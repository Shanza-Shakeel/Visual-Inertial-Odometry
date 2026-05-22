import csv
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except Exception:
    HAS_SAVGOL = False


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\Admin\vio_project_shanza")

SEQ = ROOT / "data" / "corridor3_512_16" / "dataset-corridor3_512_16"

VO_FILE = ROOT / "result" / "corridor3" / "week5_6_strong_reset_debug_corridor3" / "trajectory_tum.txt"

GT_FILE = SEQ / "mav0" / "mocap0" / "data.csv"

IMU_FILE = SEQ / "mav0" / "imu0" / "data.csv"

CALIB = SEQ / "dso" / "camchain.yaml"

IMU_CONFIG = SEQ / "dso" / "imu_config.yaml"

OUT = ROOT / "result" / "corridor3" / "week8_10_corridor3_IMU_NOISE_VIO"
OUT.mkdir(parents=True, exist_ok=True)

OUT_TRAJ_RAW = OUT / "corridor3_vio_raw_tum.txt"
OUT_TRAJ_FILTERED = OUT / "corridor3_vio_filtered_tum.txt"
OUT_METRICS = OUT / "corridor3_vio_metrics.txt"
OUT_XY = OUT / "corridor3_xy.png"
OUT_XZ = OUT / "corridor3_xz.png"


# ============================================================
# CONFIG
# ============================================================

START_FRAME_INDEX = 0
END_FRAME_INDEX = None

KEYFRAME_STEP = 120
MAX_NFEV = 15
MAX_ASSOC_DT = 0.05

TIME_OFFSETS = np.array([
    -0.04, -0.03, -0.02, -0.01,
    0.0,
    0.01, 0.02, 0.03, 0.04
])

SIGMA_VIS_POS = 0.25
SIGMA_SCALE_PRIOR = 0.75
SIGMA_ANCHOR = 1e-5
SIGMA_END_ANCHOR = 1e9
SIGMA_BA_PRIOR = 0.50
SIGMA_ROT_PRIOR = 0.12
MAX_ROT_CORR = 0.30

SIGMA_IMU_ROT_FLOOR = 0.05
SIGMA_IMU_VEL_FLOOR = 2.00
SIGMA_IMU_POS_FLOOR = 2.00

SCALE_LOW = 0.005
SCALE_HIGH = 0.20

USE_FILTER = True
FILTER_WINDOW = 17
FILTER_POLYORDER = 3

BA = np.zeros(3, dtype=np.float64)


# ============================================================
# SO3 HELPERS
# ============================================================

def skew(w):
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ], dtype=np.float64)


def exp_so3(w):
    th = np.linalg.norm(w)

    if th < 1e-12:
        return np.eye(3) + skew(w)

    K = skew(w / th)

    return (
        np.eye(3)
        + np.sin(th) * K
        + (1.0 - np.cos(th)) * (K @ K)
    )


def log_so3(R):
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(c)

    if th < 1e-12:
        return np.zeros(3)

    W = (R - R.T) / (2.0 * np.sin(th))

    return th * np.array([
        W[2, 1],
        W[0, 2],
        W[1, 0]
    ], dtype=np.float64)


def quat_to_R(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12

    x, y, z, w = q

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def R_to_quat(R):
    q = np.empty(4, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax(np.diag(R))

        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    q /= np.linalg.norm(q) + 1e-12
    return q


# ============================================================
# LOADERS
# ============================================================

def read_yaml(path):
    text = path.read_text(encoding="utf-8")

    if text.startswith("%YAML:1.0"):
        text = text.replace("%YAML:1.0", "", 1).lstrip()

    return yaml.safe_load(text)


def load_T_cam_imu():
    cam = read_yaml(CALIB)["cam0"]

    if "T_cam_imu" not in cam:
        raise RuntimeError("T_cam_imu missing in camchain.yaml")

    return np.asarray(cam["T_cam_imu"], dtype=np.float64)


def load_imu_noise_config():
    if not IMU_CONFIG.exists():
        print("WARNING: imu_config.yaml not found. Using fallback IMU noise.")
        return {
            "gyro_noise_density": 1.6968e-04,
            "gyro_random_walk": 1.9393e-05,
            "acc_noise_density": 2.0000e-03,
            "acc_random_walk": 3.0000e-03,
            "update_rate": 200.0,
        }

    data = read_yaml(IMU_CONFIG)

    return {
        "gyro_noise_density": float(data.get("gyroscope_noise_density", data.get("gyro_noise_density", 1.6968e-04))),
        "gyro_random_walk": float(data.get("gyroscope_random_walk", data.get("gyro_random_walk", 1.9393e-05))),
        "acc_noise_density": float(data.get("accelerometer_noise_density", data.get("acc_noise_density", 2.0000e-03))),
        "acc_random_walk": float(data.get("accelerometer_random_walk", data.get("acc_random_walk", 3.0000e-03))),
        "update_rate": float(data.get("update_rate", 200.0)),
    }


def imu_residual_sigmas(pim, imu_noise):
    dt = max(float(pim["dt"]), 1e-6)

    sigma_rot = imu_noise["gyro_noise_density"] * np.sqrt(dt)
    sigma_vel = imu_noise["acc_noise_density"] * np.sqrt(dt)
    sigma_pos = 0.5 * imu_noise["acc_noise_density"] * (dt ** 1.5)

    sigma_rot = max(sigma_rot, SIGMA_IMU_ROT_FLOOR)
    sigma_vel = max(sigma_vel, SIGMA_IMU_VEL_FLOOR)
    sigma_pos = max(sigma_pos, SIGMA_IMU_POS_FLOOR)

    return sigma_rot, sigma_vel, sigma_pos


def load_tum(path):
    ts = []
    p = []
    R = []

    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a = line.strip().split()

            if len(a) < 8:
                continue

            ts.append(float(a[0]))

            p.append([
                float(a[1]),
                float(a[2]),
                float(a[3])
            ])

            qx, qy, qz, qw = map(float, a[4:8])
            R.append(quat_to_R(qx, qy, qz, qw))

    return np.asarray(ts), np.asarray(p), np.asarray(R)


def load_gt(path):
    ts = []
    p = []
    R = []

    if not path.exists():
        raise FileNotFoundError(f"Missing GT:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) < 8:
                continue

            ts.append(float(row[0]) * 1e-9)

            p.append([
                float(row[1]),
                float(row[2]),
                float(row[3])
            ])

            qw = float(row[4])
            qx = float(row[5])
            qy = float(row[6])
            qz = float(row[7])

            R.append(quat_to_R(qx, qy, qz, qw))

    return np.asarray(ts), np.asarray(p), np.asarray(R)


def load_imu(path):
    ts = []
    gyro = []
    acc = []

    if not path.exists():
        raise FileNotFoundError(f"Missing IMU:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) < 7:
                continue

            ts.append(float(row[0]) * 1e-9)

            gyro.append([
                float(row[1]),
                float(row[2]),
                float(row[3])
            ])

            acc.append([
                float(row[4]),
                float(row[5]),
                float(row[6])
            ])

    return np.asarray(ts), np.asarray(gyro), np.asarray(acc)


# ============================================================
# FILTER
# ============================================================

def filter_positions(p):
    if not USE_FILTER or len(p) < FILTER_WINDOW:
        return p.copy()

    window = FILTER_WINDOW

    if window % 2 == 0:
        window += 1

    if window >= len(p):
        window = len(p) - 1 if len(p) % 2 == 0 else len(p)

    if window < 5:
        return p.copy()

    out = p.copy()

    if HAS_SAVGOL:
        for k in range(3):
            out[:, k] = savgol_filter(
                p[:, k],
                window_length=window,
                polyorder=min(FILTER_POLYORDER, window - 2),
                mode="nearest"
            )
    else:
        pad = window // 2
        pp = np.pad(p, ((pad, pad), (0, 0)), mode="edge")

        for i in range(len(p)):
            out[i] = np.mean(pp[i:i + window], axis=0)

    keep = window // 2
    out[:keep] = p[:keep]
    out[-keep:] = p[-keep:]

    return out


# ============================================================
# IMU PREINTEGRATION
# ============================================================

def preintegrate(t0, t1, imu_ts, gyro, acc, bg):
    idx = np.where((imu_ts >= t0) & (imu_ts <= t1))[0]

    if len(idx) < 2:
        return None

    dR = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    used = 0

    for a, b in zip(idx[:-1], idx[1:]):
        dt = imu_ts[b] - imu_ts[a]

        if dt <= 0.0 or dt > 0.02:
            continue

        w = gyro[a] - bg
        aa = acc[a] - BA

        dp += dv * dt + 0.5 * (dR @ aa) * dt * dt
        dv += (dR @ aa) * dt
        dR = dR @ exp_so3(w * dt)

        used += 1

    if used < 1:
        return None

    return {
        "dt": float(t1 - t0),
        "dR": dR,
        "dv": dv,
        "dp": dp,
    }


def precompute_edges(kf_ts, imu_ts, gyro, acc, bg, td):
    edges = []

    for i in range(len(kf_ts) - 1):
        pim = preintegrate(
            kf_ts[i] + td,
            kf_ts[i + 1] + td,
            imu_ts,
            gyro,
            acc,
            bg
        )

        if pim is not None:
            edges.append((i, i + 1, pim))

    return edges


# ============================================================
# CAMERA TO IMU
# ============================================================

def camera_to_imu_quantities(p_wc, R_wc, T_cam_imu):
    R_ci = T_cam_imu[:3, :3]
    p_ci = T_cam_imu[:3, 3]

    R_wi = R_wc @ R_ci
    p_ext = np.einsum("nij,j->ni", R_wc, p_ci)

    return R_wi, p_ext


# ============================================================
# INIT
# ============================================================

def estimate_gyro_bias(kf_ts, R_wi, imu_ts, gyro, acc, td):
    def residual(bg):
        res = []

        for i in range(len(kf_ts) - 1):
            pim = preintegrate(kf_ts[i] + td, kf_ts[i + 1] + td, imu_ts, gyro, acc, bg)

            if pim is None:
                continue

            R_vis = R_wi[i].T @ R_wi[i + 1]
            r = log_so3(pim["dR"].T @ R_vis)
            res.extend(r)

        if len(res) == 0:
            return np.ones(3) * 100.0

        return np.asarray(res)

    opt = least_squares(
        residual,
        np.zeros(3),
        loss="huber",
        f_scale=0.01,
        max_nfev=15
    )

    return opt.x, float(np.sqrt(np.mean(residual(opt.x) ** 2)))


def linear_init(kf_ts, kf_p, R_wi, p_ext, imu_edges):
    N = len(kf_ts)
    num_unknowns = 3 * N + 3 + 1

    A = []
    b = []

    for i, j, pim in imu_edges:
        dt = pim["dt"]

        dP_cam = kf_p[j] - kf_p[i]
        dP_ext = p_ext[j] - p_ext[i]

        if np.linalg.norm(dP_cam) < 1e-8:
            continue

        dP_imu_w = R_wi[i] @ pim["dp"]
        dV_imu_w = R_wi[i] @ pim["dv"]

        for a in range(3):
            row = np.zeros(num_unknowns)
            row[3 * i + a] = dt
            row[3 * N + a] = 0.5 * dt * dt
            row[-1] = -dP_cam[a]
            A.append(row)
            b.append(dP_ext[a] - dP_imu_w[a])

        for a in range(3):
            row = np.zeros(num_unknowns)
            row[3 * i + a] = -1.0
            row[3 * j + a] = 1.0
            row[3 * N + a] = -dt
            A.append(row)
            b.append(dV_imu_w[a])

    A = np.asarray(A)
    b = np.asarray(b)

    if len(A) < 30:
        return None

    x, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

    v = x[:3 * N].reshape(N, 3)
    g = x[3 * N:3 * N + 3]
    s = float(x[-1])

    rmse = float(np.sqrt(np.mean((A @ x - b) ** 2)))

    return {
        "scale": s,
        "gravity": g,
        "vel": v,
        "rank": rank,
        "rmse": rmse,
    }


# ============================================================
# OPTIMIZER
# ============================================================

def pack(p, v, theta, log_s, ba):
    return np.hstack([
        p.reshape(-1),
        v.reshape(-1),
        theta.reshape(-1),
        np.array([log_s]),
        ba.reshape(-1)
    ])


def unpack(x, N):
    p = x[:3 * N].reshape(N, 3)
    v = x[3 * N:6 * N].reshape(N, 3)
    theta = x[6 * N:9 * N].reshape(N, 3)
    s = float(np.exp(x[9 * N]))
    ba = x[9 * N + 1:9 * N + 4]

    return p, v, theta, s, ba


def corrected_rotations(R_wi, theta):
    return np.asarray([R_wi[i] @ exp_so3(theta[i]) for i in range(len(R_wi))])


def optimize_fixed_gravity(kf_p, R_wi, p_ext, imu_edges, init, gravity, imu_noise):
    N = len(kf_p)

    s0_raw = init["scale"]

    if not np.isfinite(s0_raw):
        return None

    s0_raw = abs(s0_raw)
    v0 = init["vel"]

    s0 = float(np.clip(s0_raw, SCALE_LOW, SCALE_HIGH))

    print(f"scale init raw(abs)={s0_raw:.6f}, clipped={s0:.6f}")

    p0 = s0 * kf_p + p_ext
    theta0 = np.zeros((N, 3), dtype=np.float64)
    ba0 = np.zeros(3, dtype=np.float64)

    x0 = pack(p0, v0, theta0, np.log(s0), ba0)

    p_anchor = p0[0].copy()
    p_end_anchor = p0[-1].copy()

    def residual(x):
        p, v, theta, s, ba = unpack(x, N)
        Rcorr = corrected_rotations(R_wi, theta)

        res = []

        res.extend((p[0] - p_anchor) / SIGMA_ANCHOR)

        if SIGMA_END_ANCHOR < 1e8:
            res.extend((p[-1] - p_end_anchor) / SIGMA_END_ANCHOR)

        res.append((np.log(s) - np.log(s0)) / SIGMA_SCALE_PRIOR)
        res.extend(ba / SIGMA_BA_PRIOR)
        res.extend((theta / SIGMA_ROT_PRIOR).reshape(-1))

        for i, j, pim in imu_edges:
            dt = pim["dt"]
            sigma_rot, sigma_vel, sigma_pos = imu_residual_sigmas(pim, imu_noise)

            dP_cam = kf_p[j] - kf_p[i]
            dP_ext = p_ext[j] - p_ext[i]

            r_vis = (p[j] - p[i]) - (s * dP_cam + dP_ext)
            res.extend(r_vis / SIGMA_VIS_POS)

            R_vis_corr = Rcorr[i].T @ Rcorr[j]
            r_R = log_so3(pim["dR"].T @ R_vis_corr)
            res.extend(r_R / sigma_rot)

            dp_corr = pim["dp"] - 0.5 * ba * dt * dt
            dv_corr = pim["dv"] - ba * dt

            r_p = (
                Rcorr[i].T @ (
                    p[j]
                    - p[i]
                    - v[i] * dt
                    - 0.5 * gravity * dt * dt
                )
                - dp_corr
            )

            r_v = (
                Rcorr[i].T @ (
                    v[j]
                    - v[i]
                    - gravity * dt
                )
                - dv_corr
            )

            res.extend(r_p / sigma_pos)
            res.extend(r_v / sigma_vel)

        return np.asarray(res)

    c0 = 0.5 * np.sum(residual(x0) ** 2)

    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)

    theta_start = 6 * N
    theta_end = 9 * N
    scale_idx = 9 * N

    lower[theta_start:theta_end] = -MAX_ROT_CORR
    upper[theta_start:theta_end] = MAX_ROT_CORR
    lower[scale_idx] = np.log(SCALE_LOW)
    upper[scale_idx] = np.log(SCALE_HIGH)
    lower[scale_idx + 1:scale_idx + 4] = -0.8
    upper[scale_idx + 1:scale_idx + 4] = 0.8

    opt = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.0,
        max_nfev=MAX_NFEV,
        x_scale="jac",
        verbose=1
    )

    c1 = 0.5 * np.sum(residual(opt.x) ** 2)
    p, v, theta, s, ba = unpack(opt.x, N)

    return {
        "p": p,
        "v": v,
        "theta": theta,
        "Rcorr": corrected_rotations(R_wi, theta),
        "scale": s,
        "ba": ba,
        "scale_init_raw_abs": s0_raw,
        "scale_init_clipped": s0,
        "gravity": gravity,
        "cost0": c0,
        "cost1": c1,
        "nfev": opt.nfev,
        "success": opt.success,
        "message": opt.message,
    }


# ============================================================
# RECONSTRUCTION
# ============================================================

def reconstruct_dense(vo_p, kf_indices, kf_p_opt, scale, p_ext_full):
    dense = np.zeros_like(vo_p)
    kptr = 0

    for i in range(len(vo_p)):
        while kptr + 1 < len(kf_indices) and i >= kf_indices[kptr + 1]:
            kptr += 1

        idx = kf_indices[kptr]

        p_wi = (
            kf_p_opt[kptr]
            + scale * (vo_p[i] - vo_p[idx])
            + (p_ext_full[i] - p_ext_full[idx])
        )

        dense[i] = p_wi - p_ext_full[i]

    return dense


def reconstruct_dense_rotations(vo_R, kf_indices, theta_kf):
    dense_R = np.empty_like(vo_R)
    kptr = 0

    for i in range(len(vo_R)):
        while kptr + 1 < len(kf_indices) and i >= kf_indices[kptr + 1]:
            kptr += 1

        dense_R[i] = vo_R[i] @ exp_so3(theta_kf[kptr])

    return dense_R


# ============================================================
# EVALUATION
# ============================================================

def nearest_pose(ts, p, target_t):
    idx = int(np.argmin(np.abs(ts - target_t)))
    return p[idx], ts[idx]


def evaluate_start_end(est_ts, est_p, gt_ts, gt_p):
    gt_start_p = gt_p[0]
    gt_end_p = gt_p[-1]

    est_start_p, est_start_t = nearest_pose(est_ts, est_p, gt_ts[0])
    est_end_p, est_end_t = nearest_pose(est_ts, est_p, gt_ts[-1])

    est_delta = est_end_p - est_start_p
    gt_delta = gt_end_p - gt_start_p

    scale = np.linalg.norm(gt_delta) / (np.linalg.norm(est_delta) + 1e-12)
    drift_vec = scale * est_delta - gt_delta
    drift = float(np.linalg.norm(drift_vec))
    gt_len = float(np.linalg.norm(gt_delta))
    drift_percent = 100.0 * drift / (gt_len + 1e-12)

    return {
        "scale": scale,
        "drift": drift,
        "drift_percent": drift_percent,
        "gt_length": gt_len,
        "start_dt": abs(est_start_t - gt_ts[0]),
        "end_dt": abs(est_end_t - gt_ts[-1]),
    }


# ============================================================
# SAVE / PLOT
# ============================================================

def save_tum(path, ts, p, R):
    with open(path, "w", encoding="utf-8") as f:
        for t, pp, RR in zip(ts, p, R):
            q = R_to_quat(RR)
            f.write(
                f"{t:.9f} "
                f"{pp[0]:.6f} {pp[1]:.6f} {pp[2]:.6f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def fix_visualization_frame(p):
    """
    Visualization-only correction.
    It does not modify the VIO optimization or saved trajectory.
    """

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    x_new = x
    y_new = -z
    z_new = y

    p_new = np.column_stack([x_new, y_new, z_new])

    theta = np.deg2rad(-90)

    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    return (Rz @ p_new.T).T


def plot_results(vo_p, vio_p, gt_p, title):
    vo0 = vo_p - vo_p[0]
    vio0 = vio_p - vio_p[0]
    gt0 = gt_p - gt_p[0]

    vo0 = fix_visualization_frame(vo0)
    vio0 = fix_visualization_frame(vio0)

    plt.figure(figsize=(8, 8))
    plt.plot(vo0[:, 0], vo0[:, 1], color="blue", label="VO raw", linewidth=1.2)
    plt.plot(vio0[:, 0], vio0[:, 1], color="orange", label="VIO filtered", linewidth=1.6)
    plt.scatter(gt0[0, 0], gt0[0, 1], color="green", s=90, marker="o", label="GT start")
    plt.scatter(gt0[-1, 0], gt0[-1, 1], color="red", s=90, marker="X", label="GT end")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title + " | XY")
    plt.tight_layout()
    plt.savefig(str(OUT_XY), dpi=250)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(vo0[:, 0], vo0[:, 2], color="blue", label="VO raw", linewidth=1.2)
    plt.plot(vio0[:, 0], vio0[:, 2], color="orange", label="VIO filtered", linewidth=1.6)
    plt.scatter(gt0[0, 0], gt0[0, 2], color="green", s=90, marker="o", label="GT start")
    plt.scatter(gt0[-1, 0], gt0[-1, 2], color="red", s=90, marker="X", label="GT end")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(title + " | XZ")
    plt.tight_layout()
    plt.savefig(str(OUT_XZ), dpi=250)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def run():
    print("=== CORRIDOR3 WEEK 8-10 VIO WITH DATASET IMU NOISE ===")
    print("VIO internals unchanged. Visualization uses frame correction only.")

    T_cam_imu = load_T_cam_imu()
    imu_noise = load_imu_noise_config()

    vo_ts_all, vo_p_all, vo_R_all = load_tum(VO_FILE)
    gt_ts, gt_p, gt_R = load_gt(GT_FILE)
    imu_ts, gyro, acc = load_imu(IMU_FILE)

    print(f"VO poses loaded : {len(vo_ts_all)}")
    print(f"GT poses loaded : {len(gt_ts)}")
    print(f"IMU loaded      : {len(imu_ts)}")

    end_idx = END_FRAME_INDEX if END_FRAME_INDEX is not None else len(vo_ts_all)

    vo_ts = vo_ts_all[START_FRAME_INDEX:end_idx]
    vo_p = vo_p_all[START_FRAME_INDEX:end_idx]
    vo_R = vo_R_all[START_FRAME_INDEX:end_idx]

    if len(vo_ts) < 100:
        raise RuntimeError("Too few VO poses.")

    R_wi_full, p_ext_full = camera_to_imu_quantities(vo_p, vo_R, T_cam_imu)

    kf_indices = np.arange(0, len(vo_ts), KEYFRAME_STEP)

    if kf_indices[-1] != len(vo_ts) - 1:
        kf_indices = np.append(kf_indices, len(vo_ts) - 1)

    kf_ts = vo_ts[kf_indices]
    kf_p = vo_p[kf_indices]
    kf_R_wi = R_wi_full[kf_indices]
    kf_ext = p_ext_full[kf_indices]

    best_init = None

    print("\nTrying time offsets...")

    for td in TIME_OFFSETS:
        bg, gyro_score = estimate_gyro_bias(kf_ts, kf_R_wi, imu_ts, gyro, acc, td)
        edges = precompute_edges(kf_ts, imu_ts, gyro, acc, bg, td)
        init = linear_init(kf_ts, kf_p, kf_R_wi, kf_ext, edges)

        if init is None:
            print(f"td={td:+.3f}: init failed")
            continue

        s = init["scale"]
        gnorm = np.linalg.norm(init["gravity"])
        ok = np.isfinite(s) and 0.001 < abs(s) < 10.0 and 5.0 < gnorm < 14.0

        print(
            f"td={td:+.3f} | ok={ok} | s={s:.6f} | |g|={gnorm:.6f} | "
            f"|bg|={np.linalg.norm(bg):.6f} | gyro={gyro_score:.6f} | "
            f"lin_rmse={init['rmse']:.6f} | edges={len(edges)}"
        )

        if not ok:
            continue

        score = abs(gnorm - 9.81) + init["rmse"] + 5.0 * gyro_score

        if best_init is None or score < best_init["score"]:
            best_init = {
                "td": td,
                "bg": bg,
                "gyro_score": gyro_score,
                "edges": edges,
                "init": init,
                "score": score,
            }

    if best_init is None:
        raise RuntimeError("No valid VI initialization found.")

    gravity = 9.81 * best_init["init"]["gravity"] / (
        np.linalg.norm(best_init["init"]["gravity"]) + 1e-12
    )

    print("gravity vector:", gravity)
    print("gravity norm:", np.linalg.norm(gravity))
    print("accel bias:", opt["ba"])
    print("accel bias norm:", np.linalg.norm(opt["ba"]))
    print("gyro bias:", best_init["bg"])
    print("gyro bias norm:", np.linalg.norm(best_init["bg"]))

    print("\nRunning optimizer...")

    opt = optimize_fixed_gravity(
        kf_p,
        kf_R_wi,
        kf_ext,
        best_init["edges"],
        best_init["init"],
        gravity,
        imu_noise,
    )

    if opt is None:
        raise RuntimeError("Optimizer failed.")

    print("\nReconstructing dense trajectory...")

    dense_raw = reconstruct_dense(vo_p, kf_indices, opt["p"], opt["scale"], p_ext_full)
    dense_filtered = filter_positions(dense_raw)
    dense_R = reconstruct_dense_rotations(vo_R, kf_indices, opt["theta"])

    save_tum(OUT_TRAJ_RAW, vo_ts, dense_raw, dense_R)
    save_tum(OUT_TRAJ_FILTERED, vo_ts, dense_filtered, dense_R)

    vo_eval = evaluate_start_end(vo_ts, vo_p, gt_ts, gt_p)
    raw_eval = evaluate_start_end(vo_ts, dense_raw, gt_ts, gt_p)
    filt_eval = evaluate_start_end(vo_ts, dense_filtered, gt_ts, gt_p)

    mean_rot_corr_deg = np.mean(np.linalg.norm(opt["theta"], axis=1)) * 180.0 / np.pi

    print("\nFINAL RESULT")
    print("============")
    print(f"time offset            : {best_init['td']:+.3f} s")
    print(f"optimized scale        : {opt['scale']:.6f}")
    print(f"mean rot correction    : {mean_rot_corr_deg:.4f} deg")
    print(f"VO drift               : {vo_eval['drift']:.6f} m")
    print(f"RAW VIO drift          : {raw_eval['drift']:.6f} m")
    print(f"FILTERED VIO drift     : {filt_eval['drift']:.6f} m")
    print(f"GT start-end length    : {filt_eval['gt_length']:.6f} m")

    plot_results(
        vo_p,
        dense_filtered,
        gt_p,
        f"Corridor3 VIO IMU-noise | drift={filt_eval['drift']:.3f} m"
    )

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("CORRIDOR3 WEEK 8-10 VIO WITH DATASET IMU NOISE\n")
        f.write("================================================\n")
        f.write("VIO optimization is unchanged. Frame correction is used only for visualization.\n\n")
        f.write(f"keyframe_step: {KEYFRAME_STEP}\n")
        f.write(f"time_offset: {best_init['td']}\n")
        f.write(f"gyro_bias: {best_init['bg'].tolist()}\n")
        f.write(f"optimized_scale: {opt['scale']:.9f}\n")
        f.write(f"optimized_accel_bias: {opt['ba'].tolist()}\n")
        f.write(f"mean_rotation_correction_deg: {mean_rot_corr_deg:.9f}\n")
        f.write(f"vo_drift_m: {vo_eval['drift']:.9f}\n")
        f.write(f"raw_vio_drift_m: {raw_eval['drift']:.9f}\n")
        f.write(f"filtered_vio_drift_m: {filt_eval['drift']:.9f}\n")
        f.write(f"gt_start_end_length_m: {filt_eval['gt_length']:.9f}\n")
        f.write(f"raw_trajectory: {OUT_TRAJ_RAW}\n")
        f.write(f"filtered_trajectory: {OUT_TRAJ_FILTERED}\n")

    print("\nSaved:")
    print(OUT_TRAJ_RAW)
    print(OUT_TRAJ_FILTERED)
    print(OUT_METRICS)
    print(OUT_XY)
    print(OUT_XZ)


if __name__ == "__main__":
    run()