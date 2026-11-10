# F1 Telemetry Intelligence – Data Dictionary

Comprehensive data dictionary for the Formula 1 Telemetry and Race-Lap datasets collected during **Phase 2 (Data Ingestion)** and profiled during **Phase 3 (Data Profiling)**.

- **Season:** 2024
- **Grand Prix:** Abu Dhabi (Yas Marina Circuit)
- **Session:** Race
- **Telemetry Schema:** 26 columns, 72,476 observations
- **Race Laps Schema:** 46 columns, 1,035 records

---

## 1. Telemetry Dataset (`2024_abu_dhabi_race_telemetry`)

High-frequency time-series telemetry capturing car kinematics, control inputs, spatial positioning, and proximity metrics sampled across the opening laps of the Grand Prix.

| Column Name | Data Type | Description | Unit | Null Count | Null % | Example Values | Analytical Relevance |
|---|---|---|---|---:|---:|---|---|
| `Date` | `datetime64[ns]` | UTC timestamp of the telemetry observation. | UTC Datetime | 0 | 0.00% | `2024-12-08 13:03:35.034` | Synchronizing telemetry with external timing loops and video feeds. |
| `SessionTime` | `timedelta64[ns]` | Elapsed time from official session initialization. | Timedelta | 0 | 0.00% | `0 days 00:56:47.817` | Global session timeline alignment across drivers. |
| `DriverAhead` | `object` | Racing number of the car immediately ahead on track. | Car Number (String) | 4,019 | 5.55% | `'61'`, `'4'`, `'55'` | Proximity and traffic analysis. Missing for race leaders or at start line. |
| `DistanceToDriverAhead` | `float64` | Estimated distance to the vehicle ahead along track vector. | Metres (m) | 2,957 | 4.08% | `0.0`, `21.57`, `72.32` | Slipstreaming, dirty air, and defensive DRS proximity. Null for P1 leader. |
| `Time` | `timedelta64[ns]` | Elapsed time from the start of the current individual lap. | Timedelta | 0 | 0.00% | `0 days 00:00:00.076` | Primary intra-lap temporal index; strictly monotonically increasing. |
| `RPM` | `float64` | Internal Combustion Engine (ICE) crankshaft revolutions per minute. | RPM | 0 | 0.00% | `10000.0`, `11096.4`, `0.0` | Engine power-band usage, gear shifts, and mechanical cutoffs (e.g. PER retirement). |
| `Speed` | `float64` | Instantaneous vehicle ground speed. | km/h | 0 | 0.00% | `0.0`, `188.0`, `337.0` | Velocity profiling, acceleration dynamics, and braking zone identification. |
| `nGear` | `Int64` | Selected transmission forward gear ratio. | Integer [1–8] | 0 | 0.00% | `1`, `3`, `7`, `8` | Cornering gear selection, shift timing, and driveability evaluation. |
| `Throttle` | `float64` | Accelerator pedal application percentage. | % | 0 | 0.00% | `0.0`, `72.5`, `100.0`, `104.0` | Corner exit acceleration, traction limits, and full-throttle commitment. |
| `Brake` | `bool` | Binary indicator of friction brake line pressure activation. | Boolean | 0 | 0.00% | `False`, `True` | Braking points, threshold braking duration, and trail-braking zones. |
| `DRS` | `Int64` | Drag Reduction System rear wing activation status. | Integer [0, 1] | 0 | 0.00% | `0`, `1` | Straight-line drag reduction and overtaking advantage evaluation. |
| `Source` | `object` | Telemetry channel fusion origin. | Categorical | 0 | 0.00% | `'merged'` | Pipeline lineage indicator (FastF1/tif1 sensor interpolation stream). |
| `Distance` | `float64` | Cumulative distance covered within current lap. | Metres (m) | 0 | 0.00% | `-0.0899`, `0.0`, `5217.4` | Primary spatial alignment axis; maps telemetry directly to circuit corners. |
| `RelativeDistance` | `float64` | Normalized progress around the circuit lap. | Fraction [0.0–1.0] | 0 | 0.00% | `0.0`, `0.5196`, `1.0` | Dimensionless lap progress for spatial comparisons across drivers. |
| `Status` | `object` | Vehicle track status flag during recording. | Categorical | 0 | 0.00% | `'OnTrack'` | Filtering out-of-bounds, pitlane, or yellow flag invalidated periods. |
| `X` | `float64` | Circuit Cartesian X coordinate in world space. | Metres (m) | 0 | 0.00% | `207.97`, `1647.0`, `5800.7` | 2D/3D driving line reconstruction and track trajectory mapping. |
| `Y` | `float64` | Circuit Cartesian Y coordinate in world space. | Metres (m) | 0 | 0.00% | `2046.0`, `981.0`, `11779.1` | 2D/3D driving line reconstruction and track trajectory mapping. |
| `Z` | `float64` | Circuit Cartesian elevation Z coordinate. | Metres (m) | 0 | 0.00% | `-237.0`, `-226.7`, `-124.5` | Elevation change, gradient, and aerodynamic ride-height dynamics. |
| `AccelerationX` | `float64` | Lateral acceleration (cornering G-force). | m/s² | 0 | 0.00% | `0.0`, `1.95`, `-83.01` | Cornering load, lateral grip limits, and apex chassis balance. |
| `AccelerationY` | `float64` | Longitudinal acceleration (braking / acceleration G-force). | m/s² | 0 | 0.00% | `0.0`, `1.12`, `59.62` | Longitudinal traction, power delivery, and deceleration threshold. |
| `AccelerationZ` | `float64` | Vertical acceleration (curb strike / heave dynamics). | m/s² | 0 | 0.00% | `0.0`, `0.01`, `-92.32` | Kerb compliance, chassis bottoming out, and ride roughness. |
| `Driver` | `object` | Three-letter FIA official driver identifier. | Code (String) | 0 | 0.00% | `'VER'`, `'NOR'`, `'HAM'` | Driver partitioning, benchmarking, and cohort comparisons. |
| `LapNumber` | `int64` | Current race lap number for the driver. | Integer [1–5] | 0 | 0.00% | `1`, `2`, `5` | Lap-level grouping and evolution analysis. |
| `Year` | `int64` | Formula 1 championship calendar season. | Year (Integer) | 0 | 0.00% | `2024` | Historical partitioning and multi-season trend tracking. |
| `Event` | `object` | Official Grand Prix event designation. | String | 0 | 0.00% | `'Abu Dhabi'` | Circuit metadata mapping and track layout binding. |
| `Session` | `object` | Sporting weekend session classification. | String | 0 | 0.00% | `'Race'` | Distinguishing competitive race conditions from qualifying / practice. |

---

## 2. Race-Lap Dataset (`2024_abu_dhabi_race_laps`)

Macro-level race session records detailing timing intervals, tire compound stints, pit activity, weather conditions, and FIA race-control decisions across all 58 laps.

| Column Name | Data Type | Description | Unit | Null Count | Null % | Example Values | Analytical Relevance |
|---|---|---|---|---:|---:|---|---|
| `index` | `int64` | Original DataFrame row index preserved from source. | Integer | 0 | 0.00% | `0`, `1`, `1034` | Lineage tracking and relational joins. |
| `Time` | `timedelta64[ns]` | Session elapsed time when the lap was officially completed. | Timedelta | 0 | 0.00% | `0 days 00:58:27.597` | Race clock synchronization and event sequencing. |
| `Driver` | `category` | Three-letter driver abbreviation. | Code (String) | 0 | 0.00% | `'VER'`, `'NOR'`, `'LEC'` | Driver-level aggregation and stint tracking. |
| `DriverNumber` | `object` | Official driver permanent car competition number. | String | 0 | 0.00% | `'1'`, `'4'`, `'16'` | FIA identifier resolution and telemetry linking. |
| `LapTime` | `timedelta64[ns]` | Total elapsed lap duration. | Timedelta | 2 | 0.19% | `0 days 00:01:27.450` | Primary pace target for regression and prediction baselines. |
| `LapNumber` | `float64` | Race lap counter. | Float [1.0–58.0] | 0 | 0.00% | `1.0`, `2.0`, `58.0` | Race distance progression and stint degradation modeling. |
| `Stint` | `float64` | Tyre stint index for the driver within the race. | Float [1.0–5.0] | 0 | 0.00% | `1.0`, `2.0` | Stint-length strategy and pit stop cycle identification. |
| `PitOutTime` | `timedelta64[ns]` | Session timestamp when the driver exited the pit lane. | Timedelta | 1,007 | 97.29% | `0 days 01:20:15.120` | Out-lap identification and pit stop exit loss calculation. |
| `PitInTime` | `timedelta64[ns]` | Session timestamp when the driver entered pit lane speed limit. | Timedelta | 1,005 | 97.10% | `0 days 01:19:50.040` | In-lap identification, pit lane loss, and undercut modeling. |
| `Sector1Time` | `timedelta64[ns]` | Duration required to traverse Sector 1 timing beam. | Timedelta | 21 | 2.03% | `0 days 00:00:17.382` | Micro-sector pacing and low-speed corner performance. |
| `Sector2Time` | `timedelta64[ns]` | Duration required to traverse Sector 2 timing beam. | Timedelta | 2 | 0.19% | `0 days 00:00:36.780` | Long-straight speed trap and DRS efficiency assessment. |
| `Sector3Time` | `timedelta64[ns]` | Duration required to traverse Sector 3 timing beam. | Timedelta | 2 | 0.19% | `0 days 00:00:33.290` | Technical hotel complex cornering and traction evaluation. |
| `Sector1SessionTime`| `timedelta64[ns]`| Session timestamp at Sector 1 completion beam. | Timedelta | 25 | 2.42% | `0 days 00:57:05.199` | Cumulative timing split alignment. |
| `Sector2SessionTime`| `timedelta64[ns]`| Session timestamp at Sector 2 completion beam. | Timedelta | 2 | 0.19% | `0 days 00:57:41.979` | Cumulative timing split alignment. |
| `Sector3SessionTime`| `timedelta64[ns]`| Session timestamp at Sector 3 completion beam. | Timedelta | 2 | 0.19% | `0 days 00:58:15.269` | Cumulative timing split alignment. |
| `SpeedI1` | `float64` | Speed trap measurement at Intermediate Timing 1. | km/h | 2 | 0.19% | `285.0`, `292.0` | Sector 1 terminal straight-line speed benchmark. |
| `SpeedI2` | `float64` | Speed trap measurement at Intermediate Timing 2. | km/h | 2 | 0.19% | `315.0`, `328.0` | Sector 2 long straight top-speed benchmark. |
| `SpeedFL` | `float64` | Speed recorded crossing the start-finish line. | km/h | 32 | 3.09% | `255.0`, `262.0` | Straight-line acceleration and launch momentum. |
| `SpeedST` | `float64` | Official Speed Trap measurement point along back straight. | km/h | 2 | 0.19% | `325.0`, `337.0` | Absolute maximum top-speed comparison under DRS. |
| `IsPersonalBest` | `bool` | Flag whether lap was driver's fastest up to that point. | Boolean | 0 | 0.00% | `True`, `False` | Tracking driver pace evolution and fuel burn-off benefit. |
| `Compound` | `category` | Pirelli tyre compound specification fitted to car. | Categorical | 0 | 0.00% | `'HARD'`, `'MEDIUM'`, `'SOFT'` | Primary mechanical grip and tyre degradation feature. |
| `TyreLife` | `float64` | Cumulative laps completed on current tyre set. | Laps (Float) | 0 | 0.00% | `1.0`, `15.0`, `45.0` | Modelling tyre degradation slopes and thermal cliff. |
| `FreshTyre` | `bool` | Whether tyres were brand new (unused) prior to stint. | Boolean | 0 | 0.00% | `True`, `False` | Baseline grip offset calibration. |
| `Team` | `category` | Constructor / team name. | Categorical | 0 | 0.00% | `'McLaren'`, `'Ferrari'` | Car performance clustering and constructor comparisons. |
| `LapStartTime` | `timedelta64[ns]` | Session timestamp when the driver commenced the lap. | Timedelta | 0 | 0.00% | `0 days 00:56:47.817` | Lap duration boundary alignment. |
| `LapStartDate` | `datetime64[ns]` | UTC wall-clock timestamp when lap commenced. | UTC Datetime | 0 | 0.00% | `2024-12-08 13:03:35.034` | Real-world time synchronisation. |
| `TrackStatus` | `category` | FIA race control track flag code (e.g. green, yellow, SC).| Code | 0 | 0.00% | `'1'`, `'2'`, `'12'` | Filtering out laps affected by yellow flags or safety cars. |
| `Position` | `float64` | Driver's running track position at end of lap. | Position [1–20]| 2 | 0.19% | `1.0`, `2.0`, `11.0` | Race position tracking, clean air vs dirty air modeling. |
| `Deleted` | `boolean` | Flag indicating whether lap was deleted for track limits. | Boolean | 0 | 0.00% | `False`, `True` | Lap validity filtering. |
| `DeletedReason` | `object` | FIA Steward explanation for deleted lap time. | String | 1,008 | 97.39% | `'TRACK LIMITS AT TURN 1 LAP 9'` | Audit trail for non-representative lap times. |
| `FastF1Generated`| `bool` | Flag indicating if lap record was synthetic or inferred. | Boolean | 0 | 0.00% | `False`, `True` | Data provenance and lineage verification. |
| `IsAccurate` | `bool` | Whether lap timing is representative and devoid of anomalies.| Boolean | 0 | 0.00% | `True`, `False` | Critical filter for training ML models; excludes in/out laps. |
| `WeatherTime` | `timedelta64[ns]` | Session timestamp of matching meteorological sample. | Timedelta | 0 | 0.00% | `0 days 00:56:57.160` | Weather sync point. |
| `AirTemp` | `float64` | Ambient air temperature. | &deg;C | 0 | 0.00% | `26.8`, `27.0` | Ambient thermodynamic density and cooling influence. |
| `Humidity` | `float64` | Relative ambient atmospheric humidity. | % | 0 | 0.00% | `49.0`, `51.0` | Air density and combustion oxygen availability. |
| `Pressure` | `float64` | Barometric atmospheric pressure. | mbar | 0 | 0.00% | `1017.2`, `1017.4` | Aerodynamic downforce and air density scaling. |
| `Rainfall` | `bool` | Precipitation detection flag. | Boolean | 0 | 0.00% | `False` | Dry vs wet weather condition filtering. |
| `TrackTemp` | `float64` | Asphalt surface temperature. | &deg;C | 0 | 0.00% | `31.8`, `32.2` | Tyre operating temperature and thermal degradation rate. |
| `WindDirection` | `Int64` | Azimuth direction of ambient wind vector. | Degrees [0–360] | 0 | 0.00% | `111`, `117` | Aerodynamic straight-line tailwind vs headwind impact. |
| `WindSpeed` | `float64` | Ambient wind speed. | m/s | 0 | 0.00% | `1.3`, `2.0` | Aerodynamic stability in high-speed corners. |
| `LapTimeSeconds`| `float64` | Lap duration converted directly to floating-point seconds.| Seconds (s) | 2 | 0.19% | `98.051`, `99.510` | Convenient numeric target variable for ML regression models. |
| `ms1` | `object` | Mini-sector timing bitmap representation for Sector 1. | Bitmap String | 0 | 0.00% | `'01111'`, `'11113'` | Granular corner-level pace differentiation. |
| `ms2` | `object` | Mini-sector timing bitmap representation for Sector 2. | Bitmap String | 0 | 0.00% | `'111111111'` | Granular straight-level pace differentiation. |
| `ms3` | `object` | Mini-sector timing bitmap representation for Sector 3. | Bitmap String | 1 | 0.10% | `'113111110'` | Granular corner-level pace differentiation. |
| `drv` | `object` | Auxiliary driver code identifier. | Code (String) | 0 | 0.00% | `'VER'`, `'GAS'`, `'PER'` | Secondary driver cross-check. |
| `team` | `object` | Auxiliary team identifier. | String | 0 | 0.00% | `'Red Bull Racing'` | Secondary team cross-check. |

---

## 3. Preprocessing and Engineering Directives (for Phase 4)

1. **`DistanceToDriverAhead` Imputation Policy:**
   - **DO NOT** impute with `0.0`. A distance of `0.0` represents a physical collision.
   - For race leaders (e.g. Lando Norris), maintain as `NaN` or engineer an indicator column `IsLeadingRace = True`.
2. **`Distance` Coordinate Calibration:**
   - Lap 1 grid offset values spanning `[-0.0899m, 0.0m]` should be calibrated to `0.0m` at the start-finish line.
   - Minor sub-millimeter floating point jitter (non-monotonic steps in COL Lap 4 and PIA Lap 5) should be filtered via monotonic clamping.
3. **`Throttle` Potentiometer Scaling:**
   - Samples exceeding `100.0%` (up to `104.0%`) represent potentiometer overtravel and should be clipped to `100.0%` for standard normalization.
4. **Lap Filter for Machine Learning:**
   - Use `IsAccurate == True` and `TrackStatus == '1'` (all green) when creating training splits for lap-time regression models, filtering out safety-car affected laps and pit in/out laps.

---

## 4. Derived & Feature-Engineered Columns (Phase 4 Preprocessing)

The following companion features are produced by the Phase 4 Preprocessing Pipeline (`src/preprocessing/preprocess_pipeline.py`) and stored in `data/processed/abu_dhabi_2024/processed_telemetry.parquet` and `processed_race_laps.parquet`:

| Feature Name | Target Dataset | Data Type | Physical Definition | Mathematical Guarantee / Range | Imputation / Treatment |
|---|---|---|---|---|---|
| `Distance_Corrected` | Telemetry | `float64` | Calibrated cumulative intra-lap distance in metres | **Monotonically non-decreasing ($\Delta d \ge 0.0\text{ m}$)**; $[0.0\text{ m}, 5300.0\text{ m}]$ | Negative grid offsets clamped to $0.0\text{ m}$; sub-millimetre jitter clamped to running maximum. Preserves legitimate $\Delta d = 0.0\text{ m}$ for stationary grid holds. |
| `Distance_Anomaly_Flag` | Telemetry | `bool` | Audit flag indicating distance correction | Boolean (`True` if clamped or monotonic filter triggered) | Derived flag (188 occurrences across all laps). |
| `Throttle_Clipped` | Telemetry | `float64` | Normalised throttle pedal position | Bounded $[0.0\%, 100.0\%]$ | Mechanical overtravel potentiometer readings $(100\%, 104\%]$ clamped to $100.0\%$. |
| `Throttle_Overtravel_Flag` | Telemetry | `bool` | Audit flag for sensor overtravel | Boolean (`True` if raw throttle $> 100.0\%$) | 32 occurrences in dataset. |
| `IsBraking` | Telemetry | `bool` | Driver hydraulic brake application | Boolean (`True` if braking, `False` if coasting/accelerating) | 14,089 braking samples. |
| `Brake_Int` | Telemetry | `int64` | Binary integer indicator of braking | Discrete binary $\{0, 1\}$ | Exact integer cast of `IsBraking` for matrix algebra and ML models. |
| `IsLeadingRace` | Telemetry | `bool` | Indicator if vehicle is leading the Grand Prix | Boolean (`True` for race leader with no ahead vehicle) | True strictly for Lando Norris (`NOR`) on Laps 2–5 (2,957 samples). Eliminates artificial imputation of missing `DistanceToDriverAhead`. |
| `IsTimingLineBoundary` | Telemetry | `bool` | Proximity to start-finish timing loop | Boolean (`True` if $\text{Distance} \le 25.0\text{ m}$) | Identifies transponder transition zone for `DriverAhead` nulls. |
| `LapClassification` | Both | `object` | FIA context taxonomy for lap analytical use | String category | Categorises all 96 driver-laps: `COMPLETE_FLYING_LAP` (38), `VSC_AFFECTED_LAP` (38), `START_LAP` (19), `DNF_OR_RETIRED` (1). |
| `IsCompleteLap` | Both | `bool` | Indicates whether lap reached completion | Boolean | `False` exclusively for Perez Lap 1 crash retirement; `True` for remaining 95 laps. |
| `IsOutlierFlag` | Both | `bool` | Statistical and physical anomaly marker | Boolean | Flags engine shutdown ($\text{RPM} = 0$), throttle overtravel, kerb strikes, and sample intervals $\Delta t > 0.5\text{ s}$. |

