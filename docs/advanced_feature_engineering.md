# Advanced Feature Engineering: Methodology & Mathematics

**Project:** F1 Telemetry Intelligence  
**Phase:** Phase 6 / 6.1 – Advanced Feature Engineering & Local PostgreSQL Analytics  
**Circuit:** Yas Marina Circuit (2024 Abu Dhabi Grand Prix)  

---

## 1. Physical Domain Taxonomy & Mathematical Formulations

The Phase 6 feature engineering pipeline translates raw and resampled vehicle telemetry into scientifically grounded mechanical, aerodynamic, and kinematic features across eight domains.

### A. Vehicle Dynamics
Formula 1 vehicle kinematics are governed by longitudinal and lateral acceleration vectors:
- **Longitudinal Acceleration ($a_{\text{long}}$):**
  $$a_{\text{long}} = \frac{\Delta v}{\Delta t}$$
  where $v$ is vehicle velocity in $\text{m/s}$ ($v_{\text{km/h}} / 3.6$), and $\Delta t = \max(t_i - t_{i-1}, 0.005\text{ s})$ avoids division by zero during irregular sampling intervals.
- **Longitudinal G-Force ($g_{\text{long}}$):**
  $$g_{\text{long}} = \frac{a_{\text{long}}}{9.80665}$$
- **Lateral Acceleration ($a_{\text{lat}}$) & Lateral G-Force ($g_{\text{lat}}$):**
  $$a_{\text{lat}} = v^2 \cdot \kappa = \frac{v^2}{R}, \quad g_{\text{lat}} = \frac{a_{\text{lat}}}{9.80665}$$
  where $\kappa$ is trajectory curvature and $R$ is instantaneous radius of curvature.
- **Combined G-Force ($g_{\text{combined}}$):**
  $$g_{\text{combined}} = \sqrt{g_{\text{long}}^2 + g_{\text{lat}}^2}$$
- **Dynamic Regimes:**
  - `HEAVY_BRAKING`: $a_{\text{long}} < -4.0\text{ m/s}^2$
  - `MEDIUM_BRAKING`: $-4.0\text{ m/s}^2 \le a_{\text{long}} < -1.5\text{ m/s}^2$
  - `HARD_ACCELERATION`: $a_{\text{long}} > 3.0\text{ m/s}^2$
  - `PARTIAL_THROTTLE_ACCEL`: $0.5\text{ m/s}^2 \le a_{\text{long}} \le 3.0\text{ m/s}^2$
  - `COASTING`: $-1.5\text{ m/s}^2 \le a_{\text{long}} < 0.5\text{ m/s}^2 \land \text{Throttle} < 5\%$

### B. Track Geometry & Corner Identification
- **Coordinate System Rectification:**
  FastF1 spatial coordinates $X, Y, Z$ are natively expressed in decimeters ($0.1\text{ m}$). Applying a scale factor of $0.1$ reconciles the coordinate perimeter ($5,242.65\text{ m}$ driven racing line) with the official $5,281\text{ m}$ Yas Marina circuit survey length. Without this scaling, track length would falsely compute to $\sim 52.4\text{ km}$.
- **Heading ($\theta$):**
  $$\theta = \text{atan2}(\Delta Y, \Delta X)$$
  Discontinuities across $2\pi$ boundaries are unwrapped using continuous phase reconstruction.
- **Trajectory Curvature ($\kappa$):**
  $$\kappa = \frac{d\theta}{ds}$$
  where $ds = \sqrt{\Delta X_m^2 + \Delta Y_m^2}$, clamped to $ds \ge 0.1\text{ m}$.
- **Radius of Curvature ($R$):**
  $$R = \frac{1}{\max(|\kappa|, 10^{-6})}$$
- **Corner Detection Algorithm:**
  A corner zone is identified when $|\kappa| \ge 0.004\text{ m}^{-1}$ (radius $R \le 250\text{ m}$) sustained over a minimum arc length ($> 15\text{ m}$). Contiguous samples are clustered into discrete Corner IDs ($1 \dots 16$), exactly matching Yas Marina Circuit turn counts. Apexes are localized at the point of maximum curvature within each corner envelope.

### C. Throttle Dynamics
- **Throttle Rate of Change ($d\text{Throttle}/dt$):**
  $$\frac{d\text{Throttle}}{dt} = \frac{\Delta \text{Throttle}}{\Delta t}$$
- **Application & Release Rates:**
  $$\text{Application Rate} = \max\left(\frac{d\text{Throttle}}{dt}, 0\right), \quad \text{Release Rate} = \max\left(-\frac{d\text{Throttle}}{dt}, 0\right)$$
- **Full Throttle (WOT):**
  Boolean flag where $\text{Throttle} \ge 98.0\%$, with cumulative timer tracking continuous and aggregate full-throttle exposure.
- **Sensor Clipping:**
  Raw throttle potentiometer signals occasionally overshoot to $104\%$. Feature `Throttle_Clipped` normalizes values strictly to $[0.0, 100.0]\%$ to prevent database constraint violations while preserving the raw sensor trace in `Throttle`.

### D. Braking Dynamics
- **Braking Event Segmentation:**
  Identified when $\text{Brake\_Int} = 1$ or deceleration $a_{\text{long}} \le -2.0\text{ m/s}^2$.
- **Event Metrics:**
  - Event ID segmentation ($1 \dots N$).
  - Braking onset flag (transition from non-braking to braking).
  - Braking distance ($\Delta \text{Distance}$) and duration ($\Delta t$).
  - Event peak deceleration G-force ($\max |g_{\text{long}}|$).
  - Trail-braking release phase detection and pedal overlap ($\text{Throttle} > 5\% \land \text{Brake} = 1$).

### E. Transmission & Drivetrain Dynamics
- **Gear Shifts:** Detected via discrete transitions in forward gear integer ($n\text{Gear}_i \neq n\text{Gear}_{i-1}$).
- **Upshift / Downshift Counters:** Cumulative event tracking across each lap.
- **Drivetrain Ratio:**
  $$\text{Ratio} = \frac{\text{RPM}}{\max(v_{\text{km/h}}, 5.0)}$$
- **RPM Powerband Reference:**
  $$\text{RPM\_Band\_Pct} = \frac{\text{RPM}}{13000.0} \times 100$$
  *Analytical Clarification:* The 13,000 RPM denominator is an empirical session-specific analytical reference scale based on the observed session maximum in the dataset ($13,090\text{ RPM}$), not a universal FIA regulatory limiter (the FIA technical regulations permit up to 15,000 RPM, with fuel flow restricted above 10,500 RPM).

### F. Timing Sector Dynamics
Sector boundaries are mapped directly to official timing line locations verified against actual FastF1 sector timestamps:
- **Sector 1:** $0\text{ m} \le d < 1,200\text{ m}$ (Verified against official S1 time at $18.166\text{ s} = 1,194.73\text{ m}$)
- **Sector 2:** $1,200\text{ m} \le d < 3,520\text{ m}$ (Verified against official S2 time at $56.353\text{ s} = 3,519.10\text{ m}$)
- **Sector 3:** $3,520\text{ m} \le d \le 5,281\text{ m}$ (Final marina complex and pit straight to finish line)

### G. Lap-Level Aggregations & Cohort Demarcation
- Summarizes 96 driver-laps across pace, speed distributions (mean, max, min, std, median, percentiles), full-throttle percentage, braking time, gear shift counts, and sector duration totals.
- **Lap Classification Breakdown:**
  - `COMPLETE_FLYING_LAP`: 38 laps (Primary analysis cohort)
  - `VSC_AFFECTED_LAP`: 38 laps
  - `START_LAP`: 19 laps
  - `DNF_OR_RETIRED`: 1 lap (Sergio Perez, Lap 1, retired at $3,932.79\text{ m}$)
  - *Distinction:* Total recorded lap profiles = 96. Total eligible analysis laps = 95. Retaining all 96 profiles ensures complete auditability while maintaining strict cohort isolation during analytical queries.

---

## 2. Descriptive vs. Predictive Feature Boundaries

To prevent target leakage and data snooping in subsequent machine learning phases:
1. **Target Distinction:** Features derived contemporaneously from brake actuation (e.g., `Brake_Int`, `IsBraking`, `Braking_Event_ID`) must never be used as predictive features to forecast `IsBraking`.
2. **Causal Ordering:** All cumulative lap features (e.g., `Gear_Change_Count`, `FullThrottle_Cumulative_Pct`) strictly aggregate backward in time; no future telemetry samples are used to construct instantaneous features.
3. **Partitioning:** Models predicting lap times or corner speeds must evaluate on GroupKFold partitioned strictly by `(Driver, LapNumber)` clusters or temporal session order.

---

## 3. Canonical Datasets

Engineered outputs are persisted in both Apache Parquet (columnar, compressed) and CSV formats under `data/processed/features/`:
1. `telemetry_features.parquet` (72,476 rows $\times$ 99 columns)
2. `lap_features.parquet` (96 rows $\times$ 34 columns)
3. `sector_features.parquet` (288 rows $\times$ 15 columns)
4. `corner_features.parquet` (1,469 rows $\times$ 15 columns)
5. `feature_dictionary.csv` (58 core feature definitions across 9 metadata attributes)
