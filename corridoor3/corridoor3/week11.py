from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS - OUTDOORS5
# ============================================================

ROOT = Path(r"C:\Users\Admin\vio_project_shanza")

VO_FILE = ROOT / "result" / "outdoor5" / "week5_6_strong_reset_debug_outdoors5" / "trajectory_tum.txt"

VIO_FILE = ROOT / "result" / "outdoor5" / "week8_10_outdoors5_ALL_FAST_IMU_NOISE" / "outdoors5_vio_filtered_tum.txt"

ORB_FILE = Path(r"C:\Users\Admin\Downloads\f_outdoors5_orbslam.txt")

OUT = ROOT / "result" / "outdoor5" / "week11_final_evaluation"

OUT.mkdir(parents=True, exist_ok=True)

OUT_XY = OUT / "outdoors5_xy.png"
OUT_XZ = OUT / "outdoors5_xz.png"
OUT_YZ = OUT / "outdoors5_yz.png"
OUT_STEP = OUT / "outdoors5_steps.png"
OUT_METRICS = OUT / "outdoors5_tracking_metrics.csv"


# ============================================================
# LOAD TUM
# ============================================================

def load_tum(path):
    ts = []
    p = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            a = line.split()

            if len(a) < 8:
                continue

            ts.append(float(a[0]))
            p.append([float(a[1]), float(a[2]), float(a[3])])

    return np.array(ts), np.array(p)


# ============================================================
# VISUALIZATION FIX ONLY
# ============================================================

def fix_visualization_frame(p):
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    x_new = x
    y_new = -z
    z_new = y

    p = np.column_stack([x_new, y_new, z_new])

    theta = np.deg2rad(-90)

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    return (R @ p.T).T


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def step_displacement(p):
    if len(p) < 2:
        return np.array([])
    return np.linalg.norm(np.diff(p, axis=0), axis=1)


def path_length(p):
    steps = step_displacement(p)
    if len(steps) == 0:
        return 0.0
    return float(np.sum(steps))


def end_pose_error(est, ref):
    N = min(len(est), len(ref))
    est = est[:N]
    ref = ref[:N]
    return float(np.linalg.norm(est[-1] - ref[-1]))


def start_end_drift(p):
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(p[-1] - p[0]))


def trajectory_metrics(name, ts, p, ref=None, total_expected=None):
    steps = step_displacement(p)

    if len(ts) > 1:
        duration = float(ts[-1] - ts[0])
        dt = np.diff(ts)
        mean_dt = float(np.mean(dt))
        median_dt = float(np.median(dt))
        freq = float(1.0 / median_dt) if median_dt > 0 else np.nan
    else:
        duration = 0.0
        mean_dt = np.nan
        median_dt = np.nan
        freq = np.nan

    if total_expected is None:
        total_expected = len(p)

    tracking_coverage = 100.0 * len(p) / total_expected if total_expected > 0 else np.nan

    if ref is not None:
        epe = end_pose_error(p, ref)
    else:
        epe = np.nan

    return {
        "method": name,
        "poses_saved": len(p),
        "tracking_coverage_percent": tracking_coverage,
        "duration_s": duration,
        "mean_dt_s": mean_dt,
        "median_dt_s": median_dt,
        "approx_frequency_hz": freq,
        "path_length_m": path_length(p),
        "start_end_drift_m": start_end_drift(p),
        "end_pose_error_to_ORB_m": epe,
        "mean_step_m": float(np.mean(steps)) if len(steps) else np.nan,
        "std_step_m": float(np.std(steps)) if len(steps) else np.nan,
        "max_step_m": float(np.max(steps)) if len(steps) else np.nan,
        "median_step_m": float(np.median(steps)) if len(steps) else np.nan
    }


# ============================================================
# LOAD
# ============================================================

print("\nLoading Outdoors5 trajectories...\n")

vo_ts, vo = load_tum(VO_FILE)
vio_ts, vio = load_tum(VIO_FILE)
orb_ts, orb = load_tum(ORB_FILE)

print("Raw poses:")
print("VO:", len(vo))
print("VIO:", len(vio))
print("ORB:", len(orb))


# ============================================================
# TRACKING COVERAGE BEFORE CUTTING
# ============================================================

total_expected = max(len(vo), len(vio), len(orb))


# ============================================================
# SAME SIZE FOR VISUAL COMPARISON
# ============================================================

N = min(len(vo), len(vio), len(orb))

vo = vo[:N]
vio = vio[:N]
orb = orb[:N]

vo_ts = vo_ts[:N]
vio_ts = vio_ts[:N]
orb_ts = orb_ts[:N]


# ============================================================
# START ALIGNMENT
# ============================================================

vo = vo - vo[0]
vio = vio - vio[0]
orb = orb - orb[0]


# ============================================================
# METRICS
# ============================================================

metrics = []

metrics.append(
    trajectory_metrics(
        "VO",
        vo_ts,
        vo,
        ref=orb,
        total_expected=total_expected
    )
)

metrics.append(
    trajectory_metrics(
        "VIO",
        vio_ts,
        vio,
        ref=orb,
        total_expected=total_expected
    )
)

metrics.append(
    trajectory_metrics(
        "ORB-SLAM3",
        orb_ts,
        orb,
        ref=orb,
        total_expected=total_expected
    )
)

metrics_df = pd.DataFrame(metrics)

print("\n======================")
print("OUTDOORS5 TRACKING / TRAJECTORY METRICS")
print("======================")
print(metrics_df.to_string(index=False))

metrics_df.to_csv(OUT_METRICS, index=False)


# ============================================================
# VISUALIZATION FRAME ONLY
# ============================================================

vo_vis = fix_visualization_frame(vo)
vio_vis = fix_visualization_frame(vio)
orb_vis = orb.copy()


# ============================================================
# XY
# ============================================================

plt.figure(figsize=(9, 8))

plt.plot(vo_vis[:, 0], vo_vis[:, 1], label="VO", linewidth=2)
plt.plot(vio_vis[:, 0], vio_vis[:, 1], label="VIO", linewidth=2)
plt.plot(orb_vis[:, 0], orb_vis[:, 1], label="ORB-SLAM3", linewidth=2)

plt.scatter(0, 0, s=150, marker="o", label="Start")

plt.grid()
plt.axis("equal")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Outdoors5 XY")
plt.legend()

plt.savefig(OUT_XY, dpi=300)
plt.show()


# ============================================================
# XZ
# ============================================================

plt.figure(figsize=(9, 8))

plt.plot(vo_vis[:, 0], vo_vis[:, 2], label="VO", linewidth=2)
plt.plot(vio_vis[:, 0], vio_vis[:, 2], label="VIO", linewidth=2)
plt.plot(orb_vis[:, 0], orb_vis[:, 2], label="ORB-SLAM3", linewidth=2)

plt.scatter(0, 0, s=150, marker="o", label="Start")

plt.grid()
plt.axis("equal")
plt.xlabel("X [m]")
plt.ylabel("Z [m]")
plt.title("Outdoors5 XZ")
plt.legend()

plt.savefig(OUT_XZ, dpi=300)
plt.show()


# ============================================================
# YZ
# ============================================================

plt.figure(figsize=(9, 8))

plt.plot(vo_vis[:, 1], vo_vis[:, 2], label="VO", linewidth=2)
plt.plot(vio_vis[:, 1], vio_vis[:, 2], label="VIO", linewidth=2)
plt.plot(orb_vis[:, 1], orb_vis[:, 2], label="ORB-SLAM3", linewidth=2)

plt.scatter(0, 0, s=150, marker="o", label="Start")

plt.grid()
plt.axis("equal")
plt.xlabel("Y [m]")
plt.ylabel("Z [m]")
plt.title("Outdoors5 YZ")
plt.legend()

plt.savefig(OUT_YZ, dpi=300)
plt.show()


# ============================================================
# STEP STABILITY
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(step_displacement(vo), label="VO")
plt.plot(step_displacement(vio), label="VIO")
plt.plot(step_displacement(orb), label="ORB-SLAM3")

plt.grid()
plt.xlabel("Pose index")
plt.ylabel("Step displacement [m]")
plt.title("Outdoors5 step displacement stability")
plt.legend()

plt.savefig(OUT_STEP, dpi=300)
plt.show()


print("\n======================")
print("DONE")
print("======================")

print(f"\nSaved figures and metrics to:\n{OUT}")
print(f"\nMetrics CSV:\n{OUT_METRICS}")