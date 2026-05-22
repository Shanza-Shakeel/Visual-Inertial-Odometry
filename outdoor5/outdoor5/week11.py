from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\Admin\vio_project_shanza")

VO_FILE = (
    ROOT
    / "result"
    / "outdoor5"
    / "week5_6_strong_reset_debug_outdoors5"
    / "trajectory_tum.txt"
)

VIO_FILE = (
    ROOT
    / "result"
    / "outdoor5"
    / "week8_10_outdoors5_ALL_FAST_IMU_NOISE"
    / "outdoors5_vio_filtered_tum.txt"
)

ORB_FILE = Path(
    r"C:\Users\Admin\Downloads\f_outdoors5_orbslam.txt"
)

OUT = (
    ROOT
    / "result"
    / "outdoor5"
    / "week11_final_evaluation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OUT_XY = OUT / "xy.png"
OUT_XZ = OUT / "xz.png"
OUT_YZ = OUT / "yz.png"
OUT_STEP = OUT / "steps.png"


# ============================================================
# LOAD TUM FORMAT
# ============================================================

def load_tum(path):

    ts = []
    p = []

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            a = line.split()

            if len(a) < 8:
                continue

            ts.append(float(a[0]))

            p.append(
                [
                    float(a[1]),
                    float(a[2]),
                    float(a[3])
                ]
            )

    return np.array(ts), np.array(p)


# ============================================================
# VISUALIZATION FRAME FIX
# ============================================================

def fix_visualization_frame(p):

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    x_new = x
    y_new = -z
    z_new = y

    p = np.column_stack(
        [
            x_new,
            y_new,
            z_new
        ]
    )

    theta = np.deg2rad(-90)

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ]
    )

    p = (R @ p.T).T

    return p


# ============================================================
# STEP DISPLACEMENT
# ============================================================

def step_displacement(p):

    return np.linalg.norm(
        np.diff(
            p,
            axis=0
        ),
        axis=1
    )


# ============================================================
# END-POSE ERROR
# ============================================================

def end_pose_error(est, ref):

    return np.linalg.norm(
        est[-1] - ref[-1]
    )


# ============================================================
# LOAD TRAJECTORIES
# ============================================================

print("\nLoading trajectories...\n")

vo_ts, vo = load_tum(VO_FILE)
vio_ts, vio = load_tum(VIO_FILE)
orb_ts, orb = load_tum(ORB_FILE)


# ============================================================
# SAME SIZE
# ============================================================

N = min(
    len(vo),
    len(vio),
    len(orb)
)

vo = vo[:N]
vio = vio[:N]
orb = orb[:N]


# ============================================================
# START ALIGNMENT
# ============================================================

vo = vo - vo[0]
vio = vio - vio[0]
orb = orb - orb[0]


# ============================================================
# END-POSE ERROR
# ============================================================

vo_end_error = end_pose_error(vo, orb)
vio_end_error = end_pose_error(vio, orb)

print("\n==============================")
print("END-POSE ERROR")
print("==============================")

print(f"VO end-pose error  : {vo_end_error:.3f} m")
print(f"VIO end-pose error : {vio_end_error:.3f} m")


# ============================================================
# VISUALIZATION FRAME FIX
# ============================================================

vo_vis = fix_visualization_frame(vo)
vio_vis = fix_visualization_frame(vio)

# keep ORB unchanged
orb_vis = orb.copy()


# ============================================================
# XY PLOT
# ============================================================

plt.figure(figsize=(10, 8))

plt.plot(
    vo_vis[:, 0],
    vo_vis[:, 1],
    label="VO",
    linewidth=2
)

plt.plot(
    vio_vis[:, 0],
    vio_vis[:, 1],
    label="VIO",
    linewidth=2
)

plt.plot(
    orb_vis[:, 0],
    orb_vis[:, 1],
    label="ORB-SLAM3",
    linewidth=2
)

plt.scatter(
    0,
    0,
    s=150,
    marker="o",
    label="Start"
)

plt.grid()
plt.axis("equal")

plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.title(
    f"Outdoors5 XY | VIO end-pose drift = {vio_end_error:.3f} m"
)

plt.legend()

plt.savefig(
    OUT_XY,
    dpi=300
)

plt.show()


# ============================================================
# XZ PLOT
# ============================================================

plt.figure(figsize=(10, 8))

plt.plot(
    vo_vis[:, 0],
    vo_vis[:, 2],
    label="VO",
    linewidth=2
)

plt.plot(
    vio_vis[:, 0],
    vio_vis[:, 2],
    label="VIO",
    linewidth=2
)

plt.plot(
    orb_vis[:, 0],
    orb_vis[:, 2],
    label="ORB-SLAM3",
    linewidth=2
)

plt.scatter(
    0,
    0,
    s=150,
    marker="o",
    label="Start"
)

plt.grid()
plt.axis("equal")

plt.xlabel("X [m]")
plt.ylabel("Z [m]")

plt.title(
    f"Outdoors5 XZ | VIO end-pose drift = {vio_end_error:.3f} m"
)

plt.legend()

plt.savefig(
    OUT_XZ,
    dpi=300
)

plt.show()


# ============================================================
# YZ PLOT
# ============================================================

plt.figure(figsize=(10, 8))

plt.plot(
    vo_vis[:, 1],
    vo_vis[:, 2],
    label="VO",
    linewidth=2
)

plt.plot(
    vio_vis[:, 1],
    vio_vis[:, 2],
    label="VIO",
    linewidth=2
)

plt.plot(
    orb_vis[:, 1],
    orb_vis[:, 2],
    label="ORB-SLAM3",
    linewidth=2
)

plt.scatter(
    0,
    0,
    s=150,
    marker="o",
    label="Start"
)

plt.grid()
plt.axis("equal")

plt.xlabel("Y [m]")
plt.ylabel("Z [m]")

plt.title(
    f"Outdoors5 YZ | VIO end-pose drift = {vio_end_error:.3f} m"
)

plt.legend()

plt.savefig(
    OUT_YZ,
    dpi=300
)

plt.show()


# ============================================================
# STEP DISPLACEMENT STABILITY
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    step_displacement(vo_vis),
    label="VO"
)

plt.plot(
    step_displacement(vio_vis),
    label="VIO"
)

plt.plot(
    step_displacement(orb_vis),
    label="ORB-SLAM3"
)

plt.grid()

plt.xlabel(
    "Pose index"
)

plt.ylabel(
    "Step displacement [m]"
)

plt.title(
    "Outdoors5 step displacement stability"
)

plt.legend()

plt.savefig(
    OUT_STEP,
    dpi=300
)

plt.show()


print("\n==============================")
print("DONE")
print("==============================")

print(f"\nResults saved to:\n{OUT}")