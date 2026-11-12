import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "reports" / "feature_engineering" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
t = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features" / "telemetry_features.parquet")
c = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features" / "corner_features.parquet")

# Filter reference clean lap (VER Lap 4)
ver_tel = t[(t["Driver"] == "VER") & (t["LapNumber"] == 4)].copy()
ver_cor = c[(c["Driver"] == "VER") & (c["LapNumber"] == 4)].copy()

fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0e1117")
ax.set_facecolor("#0e1117")

# Plot track trajectory
scatter = ax.scatter(
    ver_tel["X_m"], ver_tel["Y_m"],
    c=ver_tel["Speed"], cmap="plasma",
    s=6, alpha=0.9, zorder=2
)
cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Speed (km/h)", color="white", fontsize=11)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

# Overlay detected corner apexes
for _, row in ver_cor.iterrows():
    cid = int(row["Corner_ID"])
    apex_dist = row["ApexDistance_m"]
    # Find closest telemetry row
    idx = (ver_tel["Distance"] - apex_dist).abs().idxmin()
    apex_x = ver_tel.loc[idx, "X_m"]
    apex_y = ver_tel.loc[idx, "Y_m"]
    spd = row["ApexSpeed_kmh"]
    direction = row["TurnDirection"]
    
    # Marker
    ax.scatter(apex_x, apex_y, color="#00ffcc", edgecolor="black", s=110, zorder=5)
    # Annotation
    offset_x = 35 if apex_x >= 0 else -65
    offset_y = 35 if apex_y >= 0 else -45
    ax.annotate(
        f"T{cid} ({direction[0]})\n{spd:.0f} km/h",
        (apex_x, apex_y),
        xytext=(apex_x + offset_x, apex_y + offset_y),
        textcoords="data",
        fontsize=8.5,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1f242d", edgecolor="#00ffcc", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="#00ffcc", lw=1.2),
        zorder=6
    )

ax.set_title("Yas Marina Circuit - 16 Detected Corner Apexes & Kinematic Speeds\n(Max Verstappen - Clean Flying Lap 4)",
             color="white", fontsize=14, pad=15, fontweight="bold")
ax.set_xlabel("Track Coordinate X (meters)", color="white", fontsize=11)
ax.set_ylabel("Track Coordinate Y (meters)", color="white", fontsize=11)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("#333333")

ax.axis("equal")
plt.tight_layout()

out_path = OUTPUT_DIR / "circuit_detected_corners.png"
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Saved circuit corner visualization to {out_path}")
