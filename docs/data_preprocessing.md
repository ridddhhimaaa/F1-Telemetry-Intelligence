# F1 Telemetry Intelligence – Data Preprocessing Specification (Phase 4)

---

## 1. Purpose of Phase 4

Phase 4 (**Data Validation and Preprocessing**) builds directly on the analytical evidence established during **Phase 3 (Comprehensive Data Profiling)**. Its primary mission is to transform raw, asynchronous, and boundary-affected Formula 1 telemetry and race-lap datasets into a validated, normalized, and feature-enriched analytical baseline suitable for downstream Exploratory Data Analysis (Phase 5), Feature Engineering (Phase 6), and Machine Learning (Phase 9).

### Core Principles
1. **Strict Immutability:** Source data in `data/raw/` is never modified, deleted, or overwritten.
2. **Preservation of Raw Physical Signals:** No raw records or channels are dropped. Derived representations and normalizations are stored in explicit companion features.
3. **Domain-Informed Imputation Policy:** Meaningful nulls (e.g. race leader proximity, pit timing) are preserved; zero and mean imputation are strictly avoided.
4. **Reproducibility:** The pipeline is completely deterministic and runnable via a unified CLI.

---

## 2. Input Datasets

| Dataset Entity | Format | Path | Dimensions |
|---|---|---|---|
| **Raw Telemetry** | Parquet / CSV | `data/raw/2024_abu_dhabi_race_telemetry.parquet` | 72,476 rows $\times$ 26 columns |
| **Raw Race Laps** | Parquet / CSV | `data/raw/2024_abu_dhabi_race_laps.parquet` | 1,035 rows $\times$ 46 columns |
| **Ingestion Metadata** | JSON | `data/metadata/2024_abu_dhabi_metadata.json` | 60 lines (collection parameters) |

---

## 3. Validation Rules (Stage 1)

Fourteen non-destructive validation rules are enforced across Telemetry and Race Laps:

| Rule ID | Target Entity | Expected Contract | Actual Result | Status | Severity |
|---|---|---|---|:---:|:---:|
| `VAL_TEL_COL_PRESENCE` | Telemetry Columns | All 26 required columns present | All 26 present | **PASS** | INFO |
| `VAL_LAPS_COL_PRESENCE`| Race Laps Columns | All 46 required columns present | All 46 present | **PASS** | INFO |
| `VAL_TEL_SPEED_RANGE` | Telemetry `Speed` | $0.0 \le \text{Speed} \le 360.0\text{ km/h}$ | Range: $[0.0, 337.0]\text{ km/h}$ | **PASS** | INFO |
| `VAL_TEL_THROTTLE_NOMINAL` | Telemetry `Throttle` | $0.0\% \le \text{Throttle} \le 100.0\%$ | 32 overtravel samples $(100\%, 104\%]$ | **WARNING**| WARNING |
| `VAL_TEL_BRAKE_REPRESENTATION`| Telemetry `Brake` | Boolean or binary numeric $\{0, 1\}$ | Boolean (`False`/`True`) | **PASS** | INFO |
| `VAL_TEL_RPM_RANGE` | Telemetry `RPM` | $0 < \text{RPM} \le 15000$ (operating ICE) | 33 samples with $\text{RPM} = 0$ (PER DNF) | **WARNING**| WARNING |
| `VAL_TEL_GEAR_RANGE` | Telemetry `nGear` | Integer forward gears $[1, 8]$ | Range: $[1, 8]$ (no 0 or reverse) | **PASS** | INFO |
| `VAL_TEL_DISTANCE_NON_NEGATIVE`| Telemetry `Distance` | $\text{Distance} \ge 0.0\text{ m}$ | 44 samples $< 0.0\text{ m}$ on Lap 1 grid | **WARNING**| WARNING |
| `VAL_TEL_TIME_MONOTONIC`| Telemetry `Time` | Strictly monotonic per driver-lap | 0 non-monotonic driver-laps | **PASS** | INFO |
| `VAL_LAPS_DRIVER_ROSTER`| Race Laps `Driver` | 20 official FIA driver codes | Exactly 20 drivers present | **PASS** | INFO |
| `VAL_LAPS_NUMBER_BOUNDS`| Race Laps `LapNumber`| $1 \le \text{LapNumber} \le 58$ | Min: 1, Max: 58 | **PASS** | INFO |
| `VAL_LAPS_TIME_NULLS` | Race Laps `LapTime` | Present for completed laps | 2 nulls (PER Lap 1 crash & Lap 1) | **WARNING**| WARNING |
| `VAL_CROSS_DRIVERS` | Cross-Dataset | All telemetry drivers exist in laps | 20 / 20 drivers matched (100%) | **PASS** | INFO |
| `VAL_CROSS_DRIVER_LAPS`| Cross-Dataset | All telemetry driver-laps in laps | 96 / 96 series matched (100%) | **PASS** | INFO |

---

## 4. Missing-Value Strategy (Stage 2)

Phase 3 established that missing values in this dataset are **Missing Not At Random (MNAR)** or structural:

### 1. `DistanceToDriverAhead` (2,957 Missing Values)
- **Root Cause:** All 2,957 missing records belong strictly to **Lando Norris (`NOR`)** across Laps 2, 3, 4, and 5. Norris led the race from pole position; by definition, the leader of a Grand Prix has no vehicle ahead.
- **Scientific Treatment:**
  - **NEVER** impute with `0.0` (which mathematically represents a physical impact or zero gap).
  - **NEVER** impute with column mean or median (which falsely places the race leader in heavy traffic).
  - Preserve raw `DistanceToDriverAhead` as `NaN`.
  - Derive companion boolean feature `IsLeadingRace`:
    $$\text{IsLeadingRace} = \begin{cases} \text{True} & \text{if Driver} = \text{'NOR'} \text{ and } (\text{LapNumber} \in [2, 3, 4, 5] \text{ or DistanceToDriverAhead is NaN}) \\ \text{False} & \text{otherwise} \end{cases}$$

### 2. `DriverAhead` (4,019 Missing Values)
- **Root Cause:** 3,547 nulls belong to Norris (race leader). The remaining 472 nulls belong to other drivers and cluster within the first 25.0 metres of the start-finish line boundary where timing transponder order updates across sector transitions.
- **Scientific Treatment:**
  - Preserve raw `DriverAhead` as `NaN`.
  - Derive companion boolean feature `IsTimingLineBoundary` ($\text{Distance} \le 25.0\text{ m}$).

### 3. Race Laps Structural Nulls
- `PitInTime` (1,005 nulls), `PitOutTime` (1,007 nulls): Preserved as `NaT`; cars only enter the pits 1–2 times per Grand Prix.
- `DeletedReason` (1,008 nulls): Preserved as `None`; only populated when FIA stewards delete a lap time for track limit violations.
- `LapTime` (2 nulls): Preserved as `NaT`; corresponds to Sergio Perez's Lap 1 collision retirement.

---

## 5. Distance Calibration Strategy (Stage 3)

### 1. Starting Grid Negative Offset
- **Observation:** On Lap 1, 44 telemetry observations exhibit distance values down to $-0.0899\text{ m}$.
- **Physical Reason:** Cars parked in starting grid slots behind the start-finish timing line have negative initial coordinates relative to the timing loop.
- **Correction:** Pre-line negative distance coordinates are clamped to $0.0\text{ m}$ in `Distance_Corrected`, calibrating the lap start precisely to the timing line.

### 2. Monotonic Non-Decreasing Spatial Progression
- **Observation:** 22 laps exhibited negative distance steps. Twenty were associated with Lap 1 grid-to-line resets, while two (`COL` Lap 4 and `PIA` Lap 5) exhibited sub-millimetre floating-point GPS interpolation jitter ($\Delta d \approx -0.0001\text{ m}$ to $-0.003\text{ m}$).
- **Correction:** Enforce monotonic non-decreasing spatial progression:
  $$d_{\text{corrected}, i} = \max(d_{\text{corrected}, i-1}, d_{\text{calibrated}, i})$$
- **Technical Monotonicity Guarantee:** The pipeline guarantees **monotonically non-decreasing ($\Delta d \ge 0.0\text{ m}$)** distance progression across all 96 driver-laps. It intentionally does **not** enforce strict increasing monotonicity ($\Delta d > 0.0\text{ m}$) because physical telemetry validly contains 152 samples where $\Delta d = 0.0\text{ m}$:
  1. **Stationary Grid Launch Holds (91 samples across Lap 1):** Vehicles stationary in grid slots waiting for lights out ($v = 0.0\text{ km/h}$, $\text{Throttle} = 0\%$) and pre-line negative distances clamped to $0.0\text{ m}$ create consecutive identical spatial coordinates.
  2. **Jitter Running Maximum Clamping (27 samples on COL Lap 4, 34 samples on PIA Lap 5):** Clamping negative sub-millimetre drops to the previous running maximum holds the coordinate constant for one or more cycles.
  Artificially forcing strict inequality would distort physical telemetry and fabricate synthetic motion.
- **Audit:** Raw `Distance` is preserved 100% unaltered. Any point where calibration occurred is flagged in `Distance_Anomaly_Flag = True`.

---

## 6. Throttle and Brake Signal Treatment (Stage 4)

### Throttle
- **Observation:** 32 observations exceed $100.0\%$, reaching a maximum of $104.0\%$.
- **Physical Reason:** Modern Formula 1 electronic throttle pedal travel potentiometers frequently register up to $104\%$ mechanical overtravel under maximum driver force.
- **Treatment:**
  - Raw `Throttle` is preserved unaltered.
  - Derived feature `Throttle_Clipped` = $\min(100.0, \max(0.0, \text{Throttle}))$.
  - Derived boolean audit flag `Throttle_Overtravel_Flag` = $(\text{Throttle} > 100.0)$.

### Brake
- **Observation:** Verified as a binary boolean line pressure flag (`False` / `True`).
- **Treatment:**
  - Derived feature `IsBraking` (explicit boolean).
  - Derived feature `Brake_Int` ($0$ or $1$) for matrix and regression operations.

---

## 7. Lap Classification Methodology (Stage 5)

Driver-lap sequences (96 total) are categorized into an evidence-based taxonomy:

| Classification | Count | Description | `IsCompleteLap` | Downstream Handling |
|---|---:|---|:---:|---|
| `COMPLETE_FLYING_LAP` | 38 | Green flag racing laps (Laps 4 & 5; TrackStatus 1, coverage $\ge 98\%$) | `True` | Primary cohort for high-speed benchmark and lap-time models |
| `VSC_AFFECTED_LAP` | 38 | Laps under Virtual Safety Car delta speed restriction (Laps 2 & 3; TrackStatus 26, 126, 671) | `True` | Preserved for tactical and pacing studies; excluded from flying lap ML |
| `START_LAP` | 19 | Lap 1 standing start from the grid (standing launch, cold tyres, yellow flags) | `True` | Analyzed for start launch dynamics; excluded from flying lap ML |
| `DNF_OR_RETIRED` | 1 | Sergio Perez Lap 1 collision retirement (coverage $3,932.8\text{ m} < 4,500\text{ m}$, LapTime NaT) | `False` | Isolated crash telemetry; excluded from flying lap training sets |

---

## 8. Outlier Detection Methodology (Stage 6)

Applies non-destructive domain validation and statistical IQR fencing ($3.0 \times \text{IQR}$):
1. **Zero RPM Shutdown:** 33 samples where $\text{RPM} = 0$, all belonging to Perez's Lap 1 retirement.
2. **Throttle Overtravel:** 32 samples $> 100.0\%$.
3. **High Velocity Peaks:** Terminal speeds $> 330\text{ km/h}$ under DRS slipstream.
4. **Transient Acceleration Spikes:** Kerb strikes and threshold braking deceleration exceeding $3.0 \times \text{IQR}$.
5. **Sampling Intervals:** Flagged intervals where $\Delta t > 0.500\text{ s}$.

Zero observations are removed; all anomalies are catalogued in `reports/preprocessing_report/outlier_analysis.csv` and flagged via `IsOutlierFlag`.

---

## 9. Spatial Resampling Methodology (Stage 7)

- **Purpose:** Synchronize asynchronous time-domain telemetry onto identical spatial coordinates to enable direct driver-to-driver delta overlays (e.g. Turn 5 apex speed, Turn 6 braking point).
- **Target Grid:** Uniform 1.0-metre intervals from $0.0\text{ m}$ to $5,280.0\text{ m}$ (Yas Marina circuit nominal length: $5,281.0\text{ m}$).
- **Eligibility:** Restricted to valid completed racing laps (`IsCompleteLap == True` and observed distance span $\ge 4,800.0\text{ m}$). Perez Lap 1 collision retirement ($3,932.8\text{ m}$) is excluded ($96 - 1 = 95\text{ laps}$).
- **Interpolation:**
  - Continuous channels (`Speed`, `Throttle_Clipped`, `RPM`, coordinates `X`, `Y`, `Z`): 1D linear interpolation.
  - Discrete channels (`nGear`, `IsBraking`, `DRS`): Nearest-neighbor step interpolation.
- **Verification of 495,849 Row Count vs Theoretical 501,695 Rows:**
  - A theoretical count of $501,695\text{ rows}$ assumes $95\text{ laps} \times 5,281\text{ points}$ (0.0m to 5280.0m inclusive).
  - In actual Grand Prix telemetry, drivers navigate the **racing line**, which clips apexes and minimizes turn radii, yielding an integrated lap distance between $5,215\text{ m}$ and $5,260\text{ m}$ (mean: $5,219.46\text{ m}$), rather than the track centerline ($5,281.0\text{ m}$).
  - `resample_telemetry.py` enforces a **strict anti-extrapolation rule**:
    $$\text{grid\_start} = \lceil \max(0.0, d_{\min}) \rceil, \quad \text{grid\_end} = \lfloor \min(5281.0, d_{\max}) \rfloor$$
  - Consequently, the resampler terminates interpolation when observed telemetry ends, avoiding the fabrication of $\sim 61.5\text{ m}$ of synthetic data per lap past the finish line. The resulting 495,849 spatial rows (deficit of $5,846\text{ rows}$ across 95 laps) is physically and mathematically sound.
- **Deliverables:**
  - `data/processed/abu_dhabi_2024/resampled_telemetry.parquet` (495,849 spatial slices across 95 driver-laps).
  - `reports/preprocessing_report/resampling_audit.csv` (complete 96-sequence per-lap spatial audit).

---

## 10. Output Datasets Inventory

### Processed Datasets (`data/processed/abu_dhabi_2024/`)
- `validated_telemetry.parquet`: Clean schema-validated telemetry baseline (72,476 rows $\times$ 26 cols).
- `validated_race_laps.parquet`: Clean schema-validated race-lap baseline (1,035 rows $\times$ 46 cols).
- `processed_telemetry.parquet`: Feature-engineered telemetry dataset (72,476 rows $\times$ 37 cols).
- `processed_race_laps.parquet`: Feature-engineered race-lap dataset (1,035 rows $\times$ 48 cols).
- `lap_classification.csv`: Complete classification metadata for all 96 driver-laps.
- `resampled_telemetry.parquet`: Uniform 1.0m spatial slices (495,849 rows $\times$ 14 cols).
- `preprocessing_metadata.json`: Pipeline execution parameters and before/after statistics.

### Preprocessing Reports (`reports/preprocessing_report/`)
1. `validation_summary.csv`
2. `schema_validation.csv`
3. `missing_value_treatment.csv`
4. `distance_anomaly_report.csv`
5. `signal_validation_report.csv`
6. `lap_classification_report.csv`
7. `outlier_analysis.csv`
8. `preprocessing_changes.csv`
9. `data_quality_before_after.csv`
10. `preprocessing_metadata.json`
11. `preprocessing_report.html`
12. `resampling_audit.csv`
13. `phase4_audit_report.html`

---

## 11. Execution Instructions

The preprocessing pipeline can be executed from the project root using the project virtual environment:

```powershell
# Run complete preprocessing pipeline with default Parquet inputs and spatial resampling
.\.venv\Scripts\python.exe src/preprocessing/preprocess_pipeline.py

# Run with verbose debug logging
.\.venv\Scripts\python.exe src/preprocessing/preprocess_pipeline.py --verbose

# Run with CSV format preference
.\.venv\Scripts\python.exe src/preprocessing/preprocess_pipeline.py --format csv

# Run test suite
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

---

## 12. Known Limitations

1. **Telemetry Ingestion Window:** Telemetry is capped at the first 5 laps per driver from Phase 2. Full race duration (58 laps) telemetry is not yet ingested.
2. **Early Race Conditions:** Laps 1–3 contain standing start dynamics and Virtual Safety Car restrictions. Only Laps 4 and 5 reflect green-flag race pace.
3. **Perez Crash Telemetry:** Perez has only 1 partial lap (3,932.8m). It is segmented as `DNF_OR_RETIRED` and must not be used for flying-lap benchmark models.

---

## 13. Data Integrity Verification

SHA-256 cryptographic hashes verify that raw datasets were 100% untouched during Phase 4:

- `2024_abu_dhabi_race_laps.csv`: `cd92d0a8534374606002b1b39dc51e66185b63fce79f4d67b03cd9ef05abf52c`
- `2024_abu_dhabi_race_laps.parquet`: `8f56a5e246e981011090d6a969664f06e1d2915013239db21b94a0e1843be31e`
- `2024_abu_dhabi_race_telemetry.csv`: `2104ff24209460a29e821336c73b47ab9310d14d82452fc4bf952af316d09296`
- `2024_abu_dhabi_race_telemetry.parquet`: `6fa2d79b98a92741964b03e9cbcfc347928310384e052463b6eef20bf1cde2aa`
