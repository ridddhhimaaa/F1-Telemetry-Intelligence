"""
F1 Telemetry Intelligence - Feature Engineering Pipeline Orchestrator (Phase 6)
==============================================================================

Orchestrates the entire advanced feature engineering pipeline:
1. Loads canonical processed telemetry and race laps datasets.
2. Sequentially executes modular feature engineering:
   - Vehicle Dynamics (Longitudinal/Lateral acceleration, G-forces, regimes, durations)
   - Track Geometry (Heading, curvature, radius, corner zones, apex points, turn direction)
   - Throttle Dynamics (Rates, WOT duration, partial throttle duration, transitions)
   - Braking Dynamics (Onset, duration, distance, peak G, release phase, pedal overlap)
   - Corner Performance (Entry/apex/exit speeds, speed loss, exit drive, corner durations)
   - Gear and RPM (Drivetrain ratio, shift events, up/down counts, RPM rates)
   - Timing Sectors (Sector IDs, distance in sector, sector-level aggregates)
   - Lap Aggregates (Lap-level profile summary across all 96 driver-laps)
3. Generates complete Feature Dictionary with units, formulas, and data types.
4. Saves datasets in Parquet and CSV formats under data/processed/features/.
5. Runs rigorous data quality, schema, and consistency checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.data_loader import (
    load_all_datasets,
    load_processed_telemetry,
    load_processed_laps
)
from src.features.vehicle_dynamics import compute_vehicle_dynamics_features
from src.features.track_geometry import compute_track_geometry_features
from src.features.throttle_features import compute_throttle_features
from src.features.braking_features import compute_braking_features
from src.features.corner_features import compute_corner_features
from src.features.gear_rpm_features import compute_gear_rpm_features
from src.features.sector_features import compute_sector_features
from src.features.lap_features import aggregate_lap_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("f1_telemetry.features.feature_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features"


def build_feature_dictionary() -> pd.DataFrame:
    """
    Constructs the canonical feature dictionary documenting all engineered features.
    """
    definitions = [
        # Vehicle Dynamics
        {
            "Feature_Name": "Accel_Longitudinal_mps2",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "m/s^2",
            "Formula": "dv_mps / dt (Savitzky-Golay smoothed)",
            "Source_Columns": "Speed, Time",
            "Missing_Treatment": "Clamped dt >= 0.005s; forward/backward difference at boundary",
            "Description": "Body-frame longitudinal acceleration along trajectory; negative represents deceleration.",
            "Known_Limitations": "Derived from wheel/GPS speed derivatives, not direct body-frame IMU sensor."
        },
        {
            "Feature_Name": "G_Longitudinal",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "g",
            "Formula": "Accel_Longitudinal_mps2 / 9.80665",
            "Source_Columns": "Accel_Longitudinal_mps2",
            "Missing_Treatment": "Direct conversion",
            "Description": "Longitudinal acceleration expressed in multiples of standard gravity (g).",
            "Known_Limitations": "Assumes flat track without banking pitch adjustments."
        },
        {
            "Feature_Name": "SpeedChangeRate_kmh_per_s",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "km/h/s",
            "Formula": "Accel_Longitudinal_mps2 * 3.6",
            "Source_Columns": "Accel_Longitudinal_mps2",
            "Missing_Treatment": "Direct scaling",
            "Description": "Rate of speed change in km/h per second.",
            "Known_Limitations": "Linear velocity derivative."
        },
        {
            "Feature_Name": "Accel_Lateral_mps2",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "m/s^2",
            "Formula": "(v_mps)^2 * Curvature_m_inv",
            "Source_Columns": "Speed, Curvature_m_inv",
            "Missing_Treatment": "Clipped to +/- 65 m/s^2 (+/- 6.6g)",
            "Description": "Centripetal lateral acceleration in horizontal vehicle plane.",
            "Known_Limitations": "Assumes zero lateral slip angle (ideal track path tracking)."
        },
        {
            "Feature_Name": "G_Lateral",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "g",
            "Formula": "Accel_Lateral_mps2 / 9.80665",
            "Source_Columns": "Accel_Lateral_mps2",
            "Missing_Treatment": "Direct conversion",
            "Description": "Lateral centripetal acceleration in g-force units.",
            "Known_Limitations": "Signed by direction (positive = left turn, negative = right turn)."
        },
        {
            "Feature_Name": "G_Combined",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "g",
            "Formula": "sqrt(G_Longitudinal^2 + G_Lateral^2)",
            "Source_Columns": "G_Longitudinal, G_Lateral",
            "Missing_Treatment": "Non-negative magnitude",
            "Description": "Total horizontal resultant acceleration representing tire friction circle utilization.",
            "Known_Limitations": "Does not account for vertical aerodynamic downforce variations directly."
        },
        {
            "Feature_Name": "IsAccelerating",
            "Category": "Vehicle Dynamics",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if Accel_Longitudinal > 0.5 m/s^2 and Throttle > 15% else 0",
            "Source_Columns": "Accel_Longitudinal_mps2, Throttle",
            "Missing_Treatment": "Binary threshold",
            "Description": "Binary indicator of positive longitudinal propulsion.",
            "Known_Limitations": "Excludes low-throttle coasting phases."
        },
        {
            "Feature_Name": "IsDecelerating",
            "Category": "Vehicle Dynamics",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if Accel_Longitudinal < -0.5 m/s^2 or IsBraking == 1 else 0",
            "Source_Columns": "Accel_Longitudinal_mps2, IsBraking",
            "Missing_Treatment": "Binary threshold",
            "Description": "Binary indicator of vehicle deceleration or active braking.",
            "Known_Limitations": "Includes both engine braking and friction braking."
        },
        {
            "Feature_Name": "IsCoasting",
            "Category": "Vehicle Dynamics",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if not Accel and not Decel and Throttle < 15% and Brake == 0 else 0",
            "Source_Columns": "Accel_Longitudinal_mps2, Throttle, IsBraking",
            "Missing_Treatment": "Binary threshold",
            "Description": "Binary indicator of neutral coasting (neither throttle nor brake applied).",
            "Known_Limitations": "Short transition states may trigger coasting flag."
        },
        {
            "Feature_Name": "DynamicsRegime",
            "Category": "Vehicle Dynamics",
            "Data_Type": "string",
            "Unit": "category",
            "Formula": "Conditional rule evaluation based on Throttle, Brake, Speed, Accel",
            "Source_Columns": "Speed, Throttle, IsBraking, Accel_Longitudinal_mps2",
            "Missing_Treatment": "Default 'TRANSITION'",
            "Description": "Categorical operational regime (HEAVY_BRAKING, MEDIUM_BRAKING, TRACTION_LIMITED_ACCEL, AERO_LIMITED_ACCEL, COASTING).",
            "Known_Limitations": "Heuristic boundary thresholds based on F1 powertrain telemetry."
        },
        {
            "Feature_Name": "AccelPhaseDuration_s",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "seconds",
            "Formula": "Cumulative duration where IsAccelerating == 1",
            "Source_Columns": "IsAccelerating, Time",
            "Missing_Treatment": "Resets to 0 on state transition",
            "Description": "Elapsed duration of current continuous acceleration event.",
            "Known_Limitations": "Resets upon gear shift pauses if throttle momentarily dips."
        },
        {
            "Feature_Name": "DecelPhaseDuration_s",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "seconds",
            "Formula": "Cumulative duration where IsDecelerating == 1",
            "Source_Columns": "IsDecelerating, Time",
            "Missing_Treatment": "Resets to 0 on state transition",
            "Description": "Elapsed duration of current continuous deceleration event.",
            "Known_Limitations": "Includes entry coast and braking combined."
        },
        {
            "Feature_Name": "Speed_Rolling_Std_1s",
            "Category": "Vehicle Dynamics",
            "Data_Type": "float64",
            "Unit": "km/h",
            "Formula": "Rolling standard deviation of Speed over ~1.0s window",
            "Source_Columns": "Speed",
            "Missing_Treatment": "min_periods=2, fillna(0.0)",
            "Description": "Rolling 1.0-second standard deviation of velocity (kinematic volatility).",
            "Known_Limitations": "Small window boundary artifacts at lap start."
        },

        # Track Geometry
        {
            "Feature_Name": "X_m",
            "Category": "Track Geometry",
            "Data_Type": "float64",
            "Unit": "meters",
            "Formula": "X * 0.1",
            "Source_Columns": "X",
            "Missing_Treatment": "Unit conversion from decimeters",
            "Description": "Calibrated world X coordinate in meters.",
            "Known_Limitations": "FastF1 raw coordinates are decimeter resolution."
        },
        {
            "Feature_Name": "Y_m",
            "Category": "Track Geometry",
            "Data_Type": "float64",
            "Unit": "meters",
            "Formula": "Y * 0.1",
            "Source_Columns": "Y",
            "Missing_Treatment": "Unit conversion from decimeters",
            "Description": "Calibrated world Y coordinate in meters.",
            "Known_Limitations": "FastF1 raw coordinates are decimeter resolution."
        },
        {
            "Feature_Name": "Heading_deg",
            "Category": "Track Geometry",
            "Data_Type": "float64",
            "Unit": "degrees",
            "Formula": "mod(degrees(arctan2(dY, dX)), 360)",
            "Source_Columns": "X_m, Y_m",
            "Missing_Treatment": "Unwrapped for derivative calculation",
            "Description": "Trajectory heading compass angle in degrees [0, 360).",
            "Known_Limitations": "Dependent on coordinate grid orientation."
        },
        {
            "Feature_Name": "HeadingChange_deg",
            "Category": "Track Geometry",
            "Data_Type": "float64",
            "Unit": "degrees",
            "Formula": "degrees(d(theta))",
            "Source_Columns": "Heading_deg",
            "Missing_Treatment": "Stepwise difference",
            "Description": "Stepwise heading angle difference between successive telemetry samples.",
            "Known_Limitations": "Subject to GPS positioning jitter if unsmoothed."
        },
        {
            "Feature_Name": "Curvature_m_inv",
            "Category": "Track Geometry",
            "Data_Type": "float64",
            "Unit": "m^-1",
            "Formula": "d(theta) / ds (clamped ds >= 0.1m)",
            "Source_Columns": "X_m, Y_m",
            "Missing_Treatment": "Clipped to physical bounds [-0.10, 0.10]",
            "Description": "Signed trajectory path curvature (positive = left turn, negative = right turn).",
            "Known_Limitations": "Sensitive to vehicle racing line variations across drivers."
        },
        {
            "Feature_Name": "Radius_of_Curvature_m",
            "Category": "Track Geometry",
            "Data_Type": "float64",
            "Unit": "meters",
            "Formula": "1.0 / max(|Curvature|, 1e-4)",
            "Source_Columns": "Curvature_m_inv",
            "Missing_Treatment": "Clipped to [5m, 10000m]",
            "Description": "Instantaneous radius of path curvature in meters.",
            "Known_Limitations": "Straightaways saturate at maximum limit (10,000m)."
        },
        {
            "Feature_Name": "Corner_ID",
            "Category": "Track Geometry",
            "Data_Type": "int64",
            "Unit": "integer ID",
            "Formula": "Curvature threshold segmentation (|kappa| >= 0.004 m^-1, length >= 20m)",
            "Source_Columns": "Curvature_m_inv, Distance_Corrected",
            "Missing_Treatment": "0 indicates straightaway",
            "Description": "Sequential corner identifier detected around the circuit (1 to 16).",
            "Known_Limitations": "Derived from geometric curvature; chicanes segment into distinct elements."
        },
        {
            "Feature_Name": "IsCornerZone",
            "Category": "Track Geometry",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if Corner_ID > 0 else 0",
            "Source_Columns": "Corner_ID",
            "Missing_Treatment": "Binary flag",
            "Description": "Binary indicator that vehicle is traversing an active corner.",
            "Known_Limitations": "Straightaways are strictly 0."
        },
        {
            "Feature_Name": "IsCornerApex",
            "Category": "Track Geometry",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 at maximum |Curvature| within corner interval",
            "Source_Columns": "Corner_ID, Curvature_m_inv",
            "Missing_Treatment": "Single peak index per corner",
            "Description": "Binary indicator for the geometric apex (minimum radius point) of the corner.",
            "Known_Limitations": "Identifies geometric apex, which may differ slightly from clipping apex."
        },
        {
            "Feature_Name": "TurnDirection",
            "Category": "Track Geometry",
            "Data_Type": "string",
            "Unit": "category",
            "Formula": "'LEFT' if mean(Curvature) > 0 else 'RIGHT' ('STRAIGHT' if Corner_ID == 0)",
            "Source_Columns": "Curvature_m_inv, Corner_ID",
            "Missing_Treatment": "Default 'STRAIGHT'",
            "Description": "Corner turning direction (LEFT, RIGHT, STRAIGHT).",
            "Known_Limitations": "Categorical label."
        },

        # Throttle Dynamics
        {
            "Feature_Name": "Throttle_Pct",
            "Category": "Throttle Dynamics",
            "Data_Type": "float64",
            "Unit": "%",
            "Formula": "Throttle_Clipped if available else Throttle",
            "Source_Columns": "Throttle_Clipped, Throttle",
            "Missing_Treatment": "Phase 4 calibrated",
            "Description": "Effective pedal throttle percentage [0.0, 100.0].",
            "Known_Limitations": "Driver demand pedal position, not physical throttle butterfly valve opening."
        },
        {
            "Feature_Name": "Throttle_dPct_per_s",
            "Category": "Throttle Dynamics",
            "Data_Type": "float64",
            "Unit": "%/s",
            "Formula": "d(Throttle) / dt",
            "Source_Columns": "Throttle_Pct, Time",
            "Missing_Treatment": "Clipped to [-600, 600]",
            "Description": "Rate of throttle pedal change in % per second.",
            "Known_Limitations": "Subject to sensor resolution steps."
        },
        {
            "Feature_Name": "Throttle_Application_Rate",
            "Category": "Throttle Dynamics",
            "Data_Type": "float64",
            "Unit": "%/s",
            "Formula": "max(Throttle_dPct_per_s, 0.0)",
            "Source_Columns": "Throttle_dPct_per_s",
            "Missing_Treatment": "Zero when releasing",
            "Description": "Throttle application rate (positive pedal depression speed).",
            "Known_Limitations": "Measures driver foot aggressiveness."
        },
        {
            "Feature_Name": "Throttle_Release_Rate",
            "Category": "Throttle Dynamics",
            "Data_Type": "float64",
            "Unit": "%/s",
            "Formula": "abs(min(Throttle_dPct_per_s, 0.0))",
            "Source_Columns": "Throttle_dPct_per_s",
            "Missing_Treatment": "Zero when applying",
            "Description": "Throttle release rate (pedal lift speed).",
            "Known_Limitations": "Measures driver lift-off speed."
        },
        {
            "Feature_Name": "IsFullThrottle",
            "Category": "Throttle Dynamics",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if Throttle >= 98.0% else 0",
            "Source_Columns": "Throttle_Pct",
            "Missing_Treatment": "Binary threshold",
            "Description": "Binary indicator for Wide-Open Throttle (WOT).",
            "Known_Limitations": "98% accounts for slight pedal mechanical slack."
        },
        {
            "Feature_Name": "FullThrottle_Duration_s",
            "Category": "Throttle Dynamics",
            "Data_Type": "float64",
            "Unit": "seconds",
            "Formula": "Cumulative duration where IsFullThrottle == 1",
            "Source_Columns": "IsFullThrottle, Time",
            "Missing_Treatment": "Resets to 0 when lifted",
            "Description": "Elapsed duration of current continuous full-throttle acceleration run.",
            "Known_Limitations": "Resets immediately if throttle lifts even momentarily."
        },
        {
            "Feature_Name": "FullThrottle_Cumulative_Pct",
            "Category": "Throttle Dynamics",
            "Data_Type": "float64",
            "Unit": "%",
            "Formula": "(Cumulative WOT time / Elapsed lap time) * 100",
            "Source_Columns": "IsFullThrottle, Time",
            "Missing_Treatment": "Running cumulative ratio",
            "Description": "Percentage of elapsed lap time spent at full throttle.",
            "Known_Limitations": "Increases along straights, decreases in technical sectors."
        },
        {
            "Feature_Name": "Throttle_Transition_Count",
            "Category": "Throttle Dynamics",
            "Data_Type": "int64",
            "Unit": "count",
            "Formula": "Cumulative count of throttle application transitions",
            "Source_Columns": "Throttle_Pct",
            "Missing_Treatment": "Monotonically increasing per lap",
            "Description": "Total number of throttle re-application events from off/partial throttle.",
            "Known_Limitations": "Higher counts indicate throttle pumping/modulation in low-traction corners."
        },

        # Braking Dynamics
        {
            "Feature_Name": "Braking_Event_ID",
            "Category": "Braking Dynamics",
            "Data_Type": "int64",
            "Unit": "integer ID",
            "Formula": "Segmentation of contiguous IsBraking == 1 intervals",
            "Source_Columns": "IsBraking",
            "Missing_Treatment": "0 indicates off-brake",
            "Description": "Sequential identifier of discrete braking zones within each lap.",
            "Known_Limitations": "Discrete integer index."
        },
        {
            "Feature_Name": "IsBrakingOnset",
            "Category": "Braking Dynamics",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 on first tick of Braking_Event_ID else 0",
            "Source_Columns": "Braking_Event_ID",
            "Missing_Treatment": "Single tick trigger",
            "Description": "Binary trigger indicating the exact moment the driver initiates braking.",
            "Known_Limitations": "Instantaneous onset flag."
        },
        {
            "Feature_Name": "BrakingDuration_s",
            "Category": "Braking Dynamics",
            "Data_Type": "float64",
            "Unit": "seconds",
            "Formula": "Elapsed time from IsBrakingOnset",
            "Source_Columns": "IsBraking, Time",
            "Missing_Treatment": "Resets to 0 off-brake",
            "Description": "Elapsed duration within the active braking zone.",
            "Known_Limitations": "0 when brake is not applied."
        },
        {
            "Feature_Name": "BrakingDistance_m",
            "Category": "Braking Dynamics",
            "Data_Type": "float64",
            "Unit": "meters",
            "Formula": "Distance_Corrected - Distance_at_onset",
            "Source_Columns": "IsBraking, Distance_Corrected",
            "Missing_Treatment": "Resets to 0 off-brake",
            "Description": "Traversed track distance (meters) under braking from onset.",
            "Known_Limitations": "Measures total braking zone length."
        },
        {
            "Feature_Name": "BrakeEvent_Peak_G",
            "Category": "Braking Dynamics",
            "Data_Type": "float64",
            "Unit": "g",
            "Formula": "Maximum longitudinal deceleration |a_min| / 9.80665 during event",
            "Source_Columns": "Braking_Event_ID, G_Longitudinal",
            "Missing_Treatment": "Event-wide constant",
            "Description": "Peak deceleration g-force achieved during the braking zone.",
            "Known_Limitations": "Attributed across the entire braking event for ML feature availability."
        },
        {
            "Feature_Name": "IsBrakeReleasePhase",
            "Category": "Braking Dynamics",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 after peak deceleration point until brake release else 0",
            "Source_Columns": "Braking_Event_ID, G_Longitudinal",
            "Missing_Treatment": "Binary flag",
            "Description": "Binary flag for trail-braking / brake pressure modulation phase after peak brake hit.",
            "Known_Limitations": "Captures driver modulation towards corner apex."
        },
        {
            "Feature_Name": "IsPedalOverlap",
            "Category": "Braking Dynamics",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if IsBraking == 1 and Throttle > 15% else 0",
            "Source_Columns": "IsBraking, Throttle_Pct",
            "Missing_Treatment": "Binary flag",
            "Description": "Binary flag indicating simultaneous throttle and brake pedal overlap.",
            "Known_Limitations": "Rare in F1; indicates aggressive trail-braking or differential stabilization."
        },
        {
            "Feature_Name": "TimeSinceBrakeRelease_s",
            "Category": "Braking Dynamics",
            "Data_Type": "float64",
            "Unit": "seconds",
            "Formula": "Time - Time_at_last_brake_release",
            "Source_Columns": "IsBraking, Time",
            "Missing_Treatment": "0 before first braking event",
            "Description": "Elapsed time in seconds since the driver released the brake pedal.",
            "Known_Limitations": "Tracks transition from braking to corner exit."
        },

        # Corner Performance
        {
            "Feature_Name": "Corner_Entry_Speed_kmh",
            "Category": "Corner Performance",
            "Data_Type": "float64",
            "Unit": "km/h",
            "Formula": "Speed at Corner_ID entry boundary",
            "Source_Columns": "Corner_ID, Speed",
            "Missing_Treatment": "0 outside corner",
            "Description": "Vehicle entry velocity when crossing into the detected corner zone.",
            "Known_Limitations": "Attributed across the corner interval."
        },
        {
            "Feature_Name": "Corner_Apex_Speed_kmh",
            "Category": "Corner Performance",
            "Data_Type": "float64",
            "Unit": "km/h",
            "Formula": "Speed at IsCornerApex == 1",
            "Source_Columns": "IsCornerApex, Speed",
            "Missing_Treatment": "0 outside corner",
            "Description": "Minimum / apex velocity achieved within the corner.",
            "Known_Limitations": "Measures minimum cornering speed capability."
        },
        {
            "Feature_Name": "Corner_Exit_Speed_kmh",
            "Category": "Corner Performance",
            "Data_Type": "float64",
            "Unit": "km/h",
            "Formula": "Speed at Corner_ID exit boundary",
            "Source_Columns": "Corner_ID, Speed",
            "Missing_Treatment": "0 outside corner",
            "Description": "Vehicle exit velocity when exiting the corner zone.",
            "Known_Limitations": "Measures corner exit momentum."
        },
        {
            "Feature_Name": "Corner_Speed_Loss_kmh",
            "Category": "Corner Performance",
            "Data_Type": "float64",
            "Unit": "km/h",
            "Formula": "Corner_Entry_Speed - Corner_Apex_Speed",
            "Source_Columns": "Corner_Entry_Speed_kmh, Corner_Apex_Speed_kmh",
            "Missing_Treatment": "0 outside corner",
            "Description": "Total velocity scrubbed off from corner entry to apex.",
            "Known_Limitations": "Higher in sharp hairpins (Turn 5)."
        },
        {
            "Feature_Name": "Corner_Exit_Accel_kmh",
            "Category": "Corner Performance",
            "Data_Type": "float64",
            "Unit": "km/h",
            "Formula": "Corner_Exit_Speed - Corner_Apex_Speed",
            "Source_Columns": "Corner_Exit_Speed_kmh, Corner_Apex_Speed_kmh",
            "Missing_Treatment": "0 outside corner",
            "Description": "Velocity gained from apex to corner exit (exit drive).",
            "Known_Limitations": "Measures traction and power delivery out of corners."
        },
        {
            "Feature_Name": "Corner_Traversal_Duration_s",
            "Category": "Corner Performance",
            "Data_Type": "float64",
            "Unit": "seconds",
            "Formula": "Time_exit - Time_entry",
            "Source_Columns": "Corner_ID, Time",
            "Missing_Treatment": "0 outside corner",
            "Description": "Total elapsed time in seconds traversing the corner zone.",
            "Known_Limitations": "Constant per corner interval."
        },
        {
            "Feature_Name": "IsPreApexZone",
            "Category": "Corner Performance",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 inside corner before apex point else 0",
            "Source_Columns": "Corner_ID, IsCornerApex",
            "Missing_Treatment": "Binary flag",
            "Description": "Binary flag for corner entry and deceleration phase.",
            "Known_Limitations": "0 at apex and on straightaways."
        },
        {
            "Feature_Name": "IsPostApexZone",
            "Category": "Corner Performance",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 inside corner after apex point else 0",
            "Source_Columns": "Corner_ID, IsCornerApex",
            "Missing_Treatment": "Binary flag",
            "Description": "Binary flag for corner exit and acceleration phase.",
            "Known_Limitations": "0 at apex and on straightaways."
        },

        # Transmission and Engine
        {
            "Feature_Name": "Ratio_RPM_per_kmh",
            "Category": "Transmission & Engine",
            "Data_Type": "float64",
            "Unit": "RPM / (km/h)",
            "Formula": "RPM / max(Speed, 5.0)",
            "Source_Columns": "RPM, Speed",
            "Missing_Treatment": "Speed clamped >= 5.0 km/h",
            "Description": "Mechanical gear ratio index (engine revolutions per km/h).",
            "Known_Limitations": "Discrete value per gear; slight variations from tire deformation/slip."
        },
        {
            "Feature_Name": "RPM_Band_Pct",
            "Category": "Transmission & Engine",
            "Data_Type": "float64",
            "Unit": "%",
            "Formula": "(RPM / 13000.0) * 100",
            "Source_Columns": "RPM",
            "Missing_Treatment": "Normalized to limiter",
            "Description": "Percentage of nominal 13,000 RPM maximum power limiter utilized.",
            "Known_Limitations": "Peak power occurs ~10,500-12,000 RPM due to fuel-flow limit."
        },
        {
            "Feature_Name": "RPM_Change_Rate_per_s",
            "Category": "Transmission & Engine",
            "Data_Type": "float64",
            "Unit": "RPM/s",
            "Formula": "d(RPM) / dt",
            "Source_Columns": "RPM, Time",
            "Missing_Treatment": "Clipped to [-15000, 15000]",
            "Description": "Rate of engine rotational speed change (RPM/s).",
            "Known_Limitations": "Negative spikes during upshifts due to instantaneous ratio drop."
        },
        {
            "Feature_Name": "IsGearShift",
            "Category": "Transmission & Engine",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if nGear[i] != nGear[i-1] else 0",
            "Source_Columns": "nGear",
            "Missing_Treatment": "Binary transition trigger",
            "Description": "Binary flag indicating a transmission gear shift tick.",
            "Known_Limitations": "Single sample pulse."
        },
        {
            "Feature_Name": "IsUpshift",
            "Category": "Transmission & Engine",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if nGear[i] > nGear[i-1] else 0",
            "Source_Columns": "nGear",
            "Missing_Treatment": "Binary transition trigger",
            "Description": "Binary flag indicating a transmission upshift.",
            "Known_Limitations": "Single sample pulse."
        },
        {
            "Feature_Name": "IsDownshift",
            "Category": "Transmission & Engine",
            "Data_Type": "int64",
            "Unit": "binary",
            "Formula": "1 if nGear[i] < nGear[i-1] else 0",
            "Source_Columns": "nGear",
            "Missing_Treatment": "Binary transition trigger",
            "Description": "Binary flag indicating a transmission downshift.",
            "Known_Limitations": "Single sample pulse."
        },
        {
            "Feature_Name": "Gear_Change_Count",
            "Category": "Transmission & Engine",
            "Data_Type": "int64",
            "Unit": "count",
            "Formula": "Cumulative count of gear shift events in lap",
            "Source_Columns": "IsGearShift",
            "Missing_Treatment": "Monotonically increasing per lap",
            "Description": "Running cumulative count of total gear shifts within the lap.",
            "Known_Limitations": "Resets to 0 at lap boundary."
        },
        {
            "Feature_Name": "Upshift_Count",
            "Category": "Transmission & Engine",
            "Data_Type": "int64",
            "Unit": "count",
            "Formula": "Cumulative count of upshift events in lap",
            "Source_Columns": "IsUpshift",
            "Missing_Treatment": "Monotonically increasing per lap",
            "Description": "Running cumulative count of upshifts within the lap.",
            "Known_Limitations": "Resets to 0 at lap boundary."
        },
        {
            "Feature_Name": "Downshift_Count",
            "Category": "Transmission & Engine",
            "Data_Type": "int64",
            "Unit": "count",
            "Formula": "Cumulative count of downshift events in lap",
            "Source_Columns": "IsDownshift",
            "Missing_Treatment": "Monotonically increasing per lap",
            "Description": "Running cumulative count of downshifts within the lap.",
            "Known_Limitations": "Resets to 0 at lap boundary."
        },

        # Sector Features
        {
            "Feature_Name": "Sector",
            "Category": "Sector Features",
            "Data_Type": "int64",
            "Unit": "integer (1, 2, 3)",
            "Formula": "Distance boundary mapping: S1 [0, 1200m], S2 (1200, 3520m], S3 (3520, 5281m]",
            "Source_Columns": "Distance_Corrected",
            "Missing_Treatment": "1, 2, or 3 based on track distance",
            "Description": "Official timing sector identifier for the Yas Marina Circuit.",
            "Known_Limitations": "Fixed track-specific distance boundaries."
        },
        {
            "Feature_Name": "Distance_In_Sector_m",
            "Category": "Sector Features",
            "Data_Type": "float64",
            "Unit": "meters",
            "Formula": "Distance_Corrected - Sector_Start_Distance",
            "Source_Columns": "Distance_Corrected, Sector",
            "Missing_Treatment": "Resets to 0 at sector boundary",
            "Description": "Distance traversed within the current sector (meters).",
            "Known_Limitations": "Local distance metric."
        }
    ]

    return pd.DataFrame(definitions)


def run_feature_pipeline(
    output_dir: Optional[Path] = None,
    save_datasets: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the full Phase 6 Feature Engineering Pipeline:
    1. Loads processed telemetry and processed race laps.
    2. Sequentially executes all 8 feature modules.
    3. Builds Feature Dictionary.
    4. Saves Parquet and CSV outputs.
    5. Performs data quality and schema validation.

    Returns:
        (telemetry_features_df, lap_features_df, sector_features_df, corner_summary_df, feature_dictionary_df)
    """
    out_dir = output_dir or FEATURES_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("STARTING PHASE 6 ADVANCED FEATURE ENGINEERING PIPELINE")
    logger.info("================================================================================")

    # 1. Load canonical Phase 4 datasets
    tel_df = load_processed_telemetry()
    laps_df = load_processed_laps()

    logger.info("Loaded canonical telemetry: %d rows x %d columns across %d drivers",
                len(tel_df), len(tel_df.columns), tel_df["Driver"].nunique())

    # Step 1: Track Geometry Features (computes coordinates, heading, curvature, corner IDs)
    logger.info("Step 1/8: Computing Track Geometry features...")
    geom_features = compute_track_geometry_features(tel_df)

    # Attach geometry columns to work DataFrame so downstream modules can consume Curvature & Corner_ID
    tel_with_geom = pd.concat([tel_df, geom_features], axis=1)

    # Step 2: Vehicle Dynamics Features (uses curvature for lateral acceleration)
    logger.info("Step 2/8: Computing Vehicle Dynamics features...")
    dyn_features = compute_vehicle_dynamics_features(
        tel_with_geom, curvature_col="Curvature_m_inv"
    )

    # Step 3: Throttle Dynamics Features
    logger.info("Step 3/8: Computing Throttle Dynamics features...")
    th_features = compute_throttle_features(tel_with_geom)

    # Step 4: Braking Dynamics Features
    logger.info("Step 4/8: Computing Braking Dynamics features...")
    # Attach longitudinal acceleration for peak braking calculation
    tel_with_dyn = pd.concat([tel_with_geom, dyn_features[["Accel_Longitudinal_mps2", "G_Longitudinal"]]], axis=1)
    brk_features = compute_braking_features(tel_with_dyn)

    # Step 5: Corner Performance Features
    logger.info("Step 5/8: Computing Corner Performance features...")
    corner_samples, corner_summary = compute_corner_features(tel_with_geom)

    # Step 6: Gear and RPM Features
    logger.info("Step 6/8: Computing Gear and RPM features...")
    gear_features = compute_gear_rpm_features(tel_with_geom)

    # Step 7: Sector Features
    logger.info("Step 7/8: Computing Timing Sector features...")
    # Attach preliminary features for rich sector aggregation
    tel_pre_sector = pd.concat([
        tel_with_geom,
        dyn_features[["Accel_Longitudinal_mps2", "G_Longitudinal"]],
        gear_features[["IsGearShift"]]
    ], axis=1)
    sector_samples, sector_summary = compute_sector_features(tel_pre_sector)

    # Combine all instantaneous sample-level features
    logger.info("Consolidating full instantaneous telemetry feature dataset...")
    engineered_telemetry = pd.concat([
        tel_df,
        geom_features,
        dyn_features,
        th_features,
        brk_features,
        corner_samples,
        gear_features,
        sector_samples
    ], axis=1)

    # Eliminate any potential duplicate column names from intermediate concats
    engineered_telemetry = engineered_telemetry.loc[:, ~engineered_telemetry.columns.duplicated()].copy()

    # Step 8: Lap-Level Aggregates
    logger.info("Step 8/8: Computing Lap-Level Aggregate profiles...")
    lap_summary = aggregate_lap_features(engineered_telemetry, laps_df=laps_df)

    # Build Feature Dictionary
    logger.info("Building canonical Feature Dictionary...")
    feat_dict = build_feature_dictionary()

    # Save datasets if requested
    if save_datasets:
        logger.info("Saving engineered feature datasets to %s...", out_dir)

        # 1. Telemetry Features
        tel_parquet_path = out_dir / "telemetry_features.parquet"
        tel_csv_path = out_dir / "telemetry_features.csv"
        engineered_telemetry.to_parquet(tel_parquet_path, index=False)
        engineered_telemetry.to_csv(tel_csv_path, index=False)
        logger.info("Saved telemetry features: %s (%d rows x %d cols)",
                    tel_parquet_path.name, len(engineered_telemetry), len(engineered_telemetry.columns))

        # 2. Lap Features
        lap_parquet_path = out_dir / "lap_features.parquet"
        lap_csv_path = out_dir / "lap_features.csv"
        lap_summary.to_parquet(lap_parquet_path, index=False)
        lap_summary.to_csv(lap_csv_path, index=False)
        logger.info("Saved lap features: %s (%d laps x %d cols)",
                    lap_parquet_path.name, len(lap_summary), len(lap_summary.columns))

        # 3. Sector Features
        sec_parquet_path = out_dir / "sector_features.parquet"
        sec_csv_path = out_dir / "sector_features.csv"
        sector_summary.to_parquet(sec_parquet_path, index=False)
        sector_summary.to_csv(sec_csv_path, index=False)
        logger.info("Saved sector features: %s (%d rows x %d cols)",
                    sec_parquet_path.name, len(sector_summary), len(sector_summary.columns))

        # 4. Corner Summary Features
        corner_parquet_path = out_dir / "corner_features.parquet"
        corner_csv_path = out_dir / "corner_features.csv"
        corner_summary.to_parquet(corner_parquet_path, index=False)
        corner_summary.to_csv(corner_csv_path, index=False)
        logger.info("Saved corner features: %s (%d corners x %d cols)",
                    corner_parquet_path.name, len(corner_summary), len(corner_summary.columns))

        # 5. Feature Dictionary
        dict_csv_path = out_dir / "feature_dictionary.csv"
        feat_dict.to_csv(dict_csv_path, index=False)
        logger.info("Saved feature dictionary: %s (%d documented features)",
                    dict_csv_path.name, len(feat_dict))

    logger.info("================================================================================")
    logger.info("PHASE 6 FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("================================================================================")

    return engineered_telemetry, lap_summary, sector_summary, corner_summary, feat_dict


if __name__ == "__main__":
    run_feature_pipeline()
