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
ROOT = Path(r"C:\Users\Shanza\Desktop\Semester 2\Visual-Inertial-Odometry")
SEQ = ROOT / "data" / "room2" / "dataset-room2_512_16"

VO_FILE = ROOT / "result" / "week5_6_strong_reset_debug" / "trajectory_tum.txt"
GT_FILE = SEQ / "mav0" / "mocap0" / "data.csv"
IMU_FILE = SEQ / "mav0" / "imu0" / "data.csv"
CALIB = SEQ / "dso" / "camchain.yaml"

OUT = ROOT / "result" / "week8_10_posegraph_vio_rotation_refined"
OUT.mkdir(parents=True, exist_ok=True)

OUT_TRAJ_RAW = OUT / "posegraph_vio_raw_tum.txt"
OUT_TRAJ_FILTERED = OUT / "posegraph_vio_filtered_tum.txt"
OUT_METRICS = OUT / "posegraph_vio_metrics.txt"
OUT_XY = OUT / "trajectory_xy_filtered.png"
OUT_XZ = OUT / "trajectory_xz_filtered.png"
OUT_ATE = OUT / "ate_over_time_filtered.png"


# ============================================================
# CONFIG
# ============================================================
START_FRAME_INDEX = 0
END_FRAME_INDEX = 2882

KEYFRAME_STEP = 50
MAX_NFEV = 35
MAX_ASSOC_DT = 0.02
RPE_DELTA = 20

TIME_OFFSETS = np.arange(-0.030, 0.031, 0.010)

# Existing weights
SIGMA_VIS_POS = 0.04
SIGMA_IMU_POS = 1.20
SIGMA_IMU_VEL = 2.00
SIGMA_SCALE_PRIOR = 0.25
SIGMA_ANCHOR = 1e-5
SIGMA_END_ANCHOR = 0.03
SIGMA_BA_PRIOR = 0.25

# New rotation refinement weights
SIGMA_IMU_ROT = 0.08       # radians; smaller = trust IMU rotation more
SIGMA_ROT_PRIOR = 0.08     # keeps rotation corrections small
MAX_ROT_CORR = 0.20        # radians per keyframe correction bound

# Tuned physical scale bounds. Not GT used directly, but a bounded prior.
SCALE_LOW = 0.068
SCALE_HIGH = 0.083

USE_FILTER = True
FILTER_WINDOW = 15
FILTER_POLYORDER = 3

BA = np.zeros(3, dtype=np.float64)

GRAVITY_MODE = "auto"
# options: "auto", "estimated_norm", "plus_y", "minus_y", "minus_z", "plus_z", "unit_minus_y"


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
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def log_so3(R):
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(c)

    if th < 1e-12:
        return np.zeros(3)

    W = (R - R.T) / (2.0 * np.sin(th))
    return th * np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


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


def load_tum(path):
    ts, p, R = [], [], []

    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a = line.split()
            if len(a) < 8:
                continue
            ts.append(float(a[0]))
            p.append([float(a[1]), float(a[2]), float(a[3])])
            qx, qy, qz, qw = map(float, a[4:8])
            R.append(quat_to_R(qx, qy, qz, qw))

    return np.asarray(ts), np.asarray(p), np.asarray(R)


def load_gt(path):
    ts, p, R = [], [], []

    if not path.exists():
        raise FileNotFoundError(f"Missing GT file:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 8:
                continue
            ts.append(float(row[0]) * 1e-9)
            p.append([float(row[1]), float(row[2]), float(row[3])])
            qw = float(row[4])
            qx = float(row[5])
            qy = float(row[6])
            qz = float(row[7])
            R.append(quat_to_R(qx, qy, qz, qw))

    return np.asarray(ts), np.asarray(p), np.asarray(R)


def load_imu(path):
    ts, gyro, acc = [], [], []

    if not path.exists():
        raise FileNotFoundError(f"Missing IMU file:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            ts.append(float(row[0]) * 1e-9)
            gyro.append([float(row[1]), float(row[2]), float(row[3])])
            acc.append([float(row[4]), float(row[5]), float(row[6])])

    return np.asarray(ts), np.asarray(gyro), np.asarray(acc)


# ============================================================
# FILTER
# ============================================================
def moving_average_filter(p, window=15):
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, len(p) - 1 if len(p) % 2 == 0 else len(p)))
    if window % 2 == 0:
        window -= 1

    pad = window // 2
    p_pad = np.pad(p, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(p)
    for i in range(len(p)):
        out[i] = np.mean(p_pad[i:i + window], axis=0)
    return out


def filter_positions(p):
    if not USE_FILTER:
        return p.copy()

    n = len(p)
    window = FILTER_WINDOW
    if window % 2 == 0:
        window += 1
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        return p.copy()
    if window % 2 == 0:
        window -= 1

    if HAS_SAVGOL and window > FILTER_POLYORDER:
        out = np.zeros_like(p)
        for k in range(3):
            out[:, k] = savgol_filter(
                p[:, k],
                window_length=window,
                polyorder=FILTER_POLYORDER,
                mode="nearest"
            )
    else:
        out = moving_average_filter(p, window=window)

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
            bg,
        )
        if pim is not None:
            edges.append((i, i + 1, pim))
    return edges


# ============================================================
# CAMERA -> IMU POSE
# ============================================================
def camera_to_imu_quantities(p_wc, R_wc, T_cam_imu):
    R_ci = T_cam_imu[:3, :3]
    p_ci = T_cam_imu[:3, 3]
    R_wi = R_wc @ R_ci
    p_ext = np.einsum("nij,j->ni", R_wc, p_ci)
    return R_wi, p_ext


# ============================================================
# INITIALIZATION
# ============================================================
def estimate_gyro_bias(kf_ts, R_wi, imu_ts, gyro, acc, td):
    def residual(bg):
        res = []
        for i in range(len(kf_ts) - 1):
            pim = preintegrate(
                kf_ts[i] + td,
                kf_ts[i + 1] + td,
                imu_ts,
                gyro,
                acc,
                bg,
            )
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
        max_nfev=30,
    )
    score = float(np.sqrt(np.mean(residual(opt.x) ** 2)))
    return opt.x, score


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


def build_gravity_candidates(g_init):
    g_normed = 9.81 * g_init / (np.linalg.norm(g_init) + 1e-12)
    candidates = {
        "estimated_norm": g_normed,
        "plus_y": np.array([0.0, 9.81, 0.0], dtype=np.float64),
        "minus_y": np.array([0.0, -9.81, 0.0], dtype=np.float64),
        "minus_z": np.array([0.0, 0.0, -9.81], dtype=np.float64),
        "plus_z": np.array([0.0, 0.0, 9.81], dtype=np.float64),
        "unit_minus_y": 9.81 * np.array([0.0, -1.0, 0.0], dtype=np.float64),
    }

    if GRAVITY_MODE == "auto":
        return candidates
    if GRAVITY_MODE not in candidates:
        raise ValueError(f"Unknown GRAVITY_MODE: {GRAVITY_MODE}")
    return {GRAVITY_MODE: candidates[GRAVITY_MODE]}


# ============================================================
# OPTIMIZER WITH ROTATION CORRECTION
# ============================================================
def pack(p, v, theta, log_s, ba):
    return np.hstack([
        p.reshape(-1),
        v.reshape(-1),
        theta.reshape(-1),
        np.array([log_s], dtype=np.float64),
        ba.reshape(-1),
    ])


def unpack(x, N):
    p = x[:3 * N].reshape(N, 3)
    v = x[3 * N:6 * N].reshape(N, 3)
    theta = x[6 * N:9 * N].reshape(N, 3)
    log_s = x[9 * N]
    s = float(np.exp(log_s))
    ba = x[9 * N + 1:9 * N + 4]
    return p, v, theta, s, ba


def corrected_rotations(R_wi, theta):
    return np.asarray([R_wi[i] @ exp_so3(theta[i]) for i in range(len(R_wi))])


def optimize_fixed_gravity(kf_p, R_wi, p_ext, imu_edges, init, gravity):
    N = len(kf_p)
    s0_raw = init["scale"]
    v0 = init["vel"]

    if not np.isfinite(s0_raw) or s0_raw <= 0:
        return None

    s0 = float(np.clip(s0_raw, SCALE_LOW, SCALE_HIGH))
    print(f"scale init raw={s0_raw:.6f}, clipped={s0:.6f}")

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

        # Gauge / endpoint constraints
        res.extend((p[0] - p_anchor) / SIGMA_ANCHOR)
        res.extend((p[-1] - p_end_anchor) / SIGMA_END_ANCHOR)

        # Priors
        res.append((np.log(s) - np.log(s0)) / SIGMA_SCALE_PRIOR)
        res.extend(ba / SIGMA_BA_PRIOR)
        res.extend((theta / SIGMA_ROT_PRIOR).reshape(-1))

        for i, j, pim in imu_edges:
            dt = pim["dt"]
            dP_cam = kf_p[j] - kf_p[i]
            dP_ext = p_ext[j] - p_ext[i]

            # Visual relative translation from VO
            r_vis = (p[j] - p[i]) - (s * dP_cam + dP_ext)
            res.extend(r_vis / SIGMA_VIS_POS)

            # IMU rotation residual: dR_imu should match corrected visual relative rotation
            R_vis_corr = Rcorr[i].T @ Rcorr[j]
            r_R = log_so3(pim["dR"].T @ R_vis_corr)
            res.extend(r_R / SIGMA_IMU_ROT)

            # Approximate accelerometer-bias correction
            dp_corr = pim["dp"] - 0.5 * ba * dt * dt
            dv_corr = pim["dv"] - ba * dt

            r_p = (
                Rcorr[i].T @ (
                    p[j] - p[i] - v[i] * dt - 0.5 * gravity * dt * dt
                )
                - dp_corr
            )

            r_v = (
                Rcorr[i].T @ (
                    v[j] - v[i] - gravity * dt
                )
                - dv_corr
            )

            res.extend(r_p / SIGMA_IMU_POS)
            res.extend(r_v / SIGMA_IMU_VEL)

        return np.asarray(res, dtype=np.float64)

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
        verbose=0,
    )

    c1 = 0.5 * np.sum(residual(opt.x) ** 2)
    p, v, theta, s, ba = unpack(opt.x, N)
    Rcorr = corrected_rotations(R_wi, theta)

    return {
        "p": p,
        "v": v,
        "theta": theta,
        "Rcorr": Rcorr,
        "scale": s,
        "ba": ba,
        "scale_init_raw": s0_raw,
        "scale_init_clipped": s0,
        "gravity": gravity.copy(),
        "cost0": c0,
        "cost1": c1,
        "nfev": opt.nfev,
        "success": opt.success,
        "message": opt.message,
    }


# ============================================================
# DENSE TRAJECTORY RECONSTRUCTION
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
    # Piecewise-constant keyframe rotation correction for RPE diagnostic.
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
def associate(est_ts, est_p, est_R, gt_ts, gt_p, gt_R):
    ts_o, ep, eR, gp, gR = [], [], [], [], []
    j = 0

    for i, t in enumerate(est_ts):
        while j + 1 < len(gt_ts) and abs(gt_ts[j + 1] - t) < abs(gt_ts[j] - t):
            j += 1

        if abs(gt_ts[j] - t) <= MAX_ASSOC_DT:
            ts_o.append(t)
            ep.append(est_p[i])
            eR.append(est_R[i])
            gp.append(gt_p[j])
            gR.append(gt_R[j])

    return np.asarray(ts_o), np.asarray(ep), np.asarray(eR), np.asarray(gp), np.asarray(gR)


def umeyama_sim3(src, dst):
    src = np.asarray(src)
    dst = np.asarray(dst)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    H = (Y.T @ X) / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = np.mean(np.sum(X ** 2, axis=1))
    s = np.trace(np.diag(S) @ D) / (var + 1e-12)
    t = mu_d - s * R @ mu_s
    return s, R, t


def umeyama_se3(src, dst):
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    H = (Y.T @ X) / len(src)
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = mu_d - R @ mu_s
    return R, t


def apply_align(p, Rarr, gp, mode):
    if mode == "sim3":
        s, R, t = umeyama_sim3(p, gp)
        pa = (s * (R @ p.T)).T + t
        Ra = np.asarray([R @ Ri for Ri in Rarr])
        return pa, Ra, s

    if mode == "se3":
        R, t = umeyama_se3(p, gp)
        pa = (R @ p.T).T + t
        Ra = np.asarray([R @ Ri for Ri in Rarr])
        return pa, Ra, 1.0

    raise ValueError("mode must be sim3 or se3")


def ate(p, gp):
    e = np.linalg.norm(p - gp, axis=1)
    return float(np.sqrt(np.mean(e ** 2))), e


def rpe(p, R, gp, gR, delta=RPE_DELTA):
    et, er = [], []
    for i in range(len(p) - delta):
        et.append(np.linalg.norm((p[i + delta] - p[i]) - (gp[i + delta] - gp[i])))
        Re = R[i].T @ R[i + delta]
        Rg = gR[i].T @ gR[i + delta]
        er.append(np.linalg.norm(log_so3(Rg.T @ Re)) * 180.0 / np.pi)
    et = np.asarray(et)
    er = np.asarray(er)
    return float(np.sqrt(np.mean(et ** 2))), float(np.sqrt(np.mean(er ** 2)))


def evaluate(est_ts, est_p, est_R, gt_ts, gt_p, gt_R, mode):
    ts, ep, eR, gp, gR = associate(est_ts, est_p, est_R, gt_ts, gt_p, gt_R)
    if len(ep) < 50:
        raise RuntimeError(f"Too few associated poses: {len(ep)}")
    pa, Ra, scale = apply_align(ep, eR, gp, mode)
    ate_rmse, ate_series = ate(pa, gp)
    rpe_t, rpe_r = rpe(pa, Ra, gp, gR)
    return {
        "ts": ts,
        "p": pa,
        "R": Ra,
        "gt_p": gp,
        "scale": scale,
        "ate": ate_rmse,
        "ate_series": ate_series,
        "rpe_t": rpe_t,
        "rpe_r": rpe_r,
        "associated": len(ep),
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


def plot_results(vio_res):
    gt = vio_res["gt_p"]
    plt.figure(figsize=(8, 8))
    plt.plot(gt[:, 0], gt[:, 1], label="GT", linewidth=2)
    plt.plot(vio_res["p"][:, 0], vio_res["p"][:, 1], label="VIO SE(3) rot-refined filtered", linewidth=2)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory XY: GT vs VIO SE(3)")
    plt.tight_layout()
    plt.savefig(str(OUT_XY), dpi=250)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(gt[:, 0], gt[:, 2], label="GT", linewidth=2)
    plt.plot(vio_res["p"][:, 0], vio_res["p"][:, 2], label="VIO SE(3) rot-refined filtered", linewidth=2)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Trajectory XZ: GT vs VIO SE(3)")
    plt.tight_layout()
    plt.savefig(str(OUT_XZ), dpi=250)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(vio_res["ts"] - vio_res["ts"][0], vio_res["ate_series"], label="VIO SE(3) rot-refined filtered")
    plt.grid(True)
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("ATE [m]")
    plt.title("VIO ATE over time")
    plt.tight_layout()
    plt.savefig(str(OUT_ATE), dpi=250)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def run():
    print("=== WEEK 8-10 POSE-GRAPH VIO WITH ROTATION REFINEMENT ===")
    print("Uses saved VO trajectory + IMU. No image rerun. No landmarks.")
    print("Adds keyframe rotation correction, accel-bias estimate, endpoint-safe filter.")

    T_cam_imu = load_T_cam_imu()
    vo_ts_all, vo_p_all, vo_R_all = load_tum(VO_FILE)
    gt_ts, gt_p, gt_R = load_gt(GT_FILE)
    imu_ts, gyro, acc = load_imu(IMU_FILE)

    print(f"VO poses loaded : {len(vo_ts_all)}")
    print(f"GT poses loaded : {len(gt_ts)}")
    print(f"IMU loaded      : {len(imu_ts)}")
    print(f"T_cam_imu:\n{T_cam_imu}")

    end_idx = END_FRAME_INDEX if END_FRAME_INDEX is not None else len(vo_ts_all)
    vo_ts = vo_ts_all[START_FRAME_INDEX:end_idx]
    vo_p = vo_p_all[START_FRAME_INDEX:end_idx]
    vo_R = vo_R_all[START_FRAME_INDEX:end_idx]

    print(f"Using VO segment index {START_FRAME_INDEX}:{end_idx}")
    print(f"Segment poses: {len(vo_ts)}")

    R_wi_full, p_ext_full = camera_to_imu_quantities(vo_p, vo_R, T_cam_imu)

    kf_indices = np.arange(0, len(vo_ts), KEYFRAME_STEP)
    if kf_indices[-1] != len(vo_ts) - 1:
        kf_indices = np.append(kf_indices, len(vo_ts) - 1)

    kf_ts = vo_ts[kf_indices]
    kf_p = vo_p[kf_indices]
    kf_R_wi = R_wi_full[kf_indices]
    kf_ext = p_ext_full[kf_indices]

    print(f"Keyframes: {len(kf_indices)}")
    print(f"KF step: {KEYFRAME_STEP}")
    print(f"Last keyframe index: {kf_indices[-1]} / {len(vo_ts)-1}")

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
        g = init["gravity"]
        gnorm = np.linalg.norm(g)
        ok = np.isfinite(s) and 0.01 < s < 0.30 and 6.0 < gnorm < 13.0

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

    print("\nBest init:")
    print(f"time offset : {best_init['td']:+.3f} s")
    print(f"scale raw   : {best_init['init']['scale']:.6f}")
    print(f"gravity raw : {best_init['init']['gravity']}")
    print(f"|g| raw     : {np.linalg.norm(best_init['init']['gravity']):.6f}")
    print(f"gyro bias   : {best_init['bg']}")
    print(f"|bg|        : {np.linalg.norm(best_init['bg']):.6f}")

    gravity_candidates = build_gravity_candidates(best_init["init"]["gravity"])
    vo_res = evaluate(vo_ts, vo_p, vo_R, gt_ts, gt_p, gt_R, mode="sim3")

    results = []
    print("\nTesting gravity candidates inside VIO optimizer...")

    for name, gravity in gravity_candidates.items():
        print(f"\n--- Gravity mode: {name} ---")
        print(f"gravity = {gravity}, norm = {np.linalg.norm(gravity):.6f}")

        opt = optimize_fixed_gravity(
            kf_p,
            kf_R_wi,
            kf_ext,
            best_init["edges"],
            best_init["init"],
            gravity,
        )

        if opt is None:
            print("optimizer failed")
            continue

        dense_raw = reconstruct_dense(vo_p, kf_indices, opt["p"], opt["scale"], p_ext_full)
        dense_filtered = filter_positions(dense_raw)
        dense_R = reconstruct_dense_rotations(vo_R, kf_indices, opt["theta"])

        raw_res = evaluate(vo_ts, dense_raw, dense_R, gt_ts, gt_p, gt_R, mode="se3")
        filtered_res = evaluate(vo_ts, dense_filtered, dense_R, gt_ts, gt_p, gt_R, mode="se3")
        diag_res = evaluate(vo_ts, dense_filtered, dense_R, gt_ts, gt_p, gt_R, mode="sim3")

        raw_ate_imp = 100.0 * (vo_res["ate"] - raw_res["ate"]) / (vo_res["ate"] + 1e-12)
        raw_rpe_imp = 100.0 * (vo_res["rpe_t"] - raw_res["rpe_t"]) / (vo_res["rpe_t"] + 1e-12)
        filt_ate_imp = 100.0 * (vo_res["ate"] - filtered_res["ate"]) / (vo_res["ate"] + 1e-12)
        filt_rpe_imp = 100.0 * (vo_res["rpe_t"] - filtered_res["rpe_t"]) / (vo_res["rpe_t"] + 1e-12)

        mean_rot_corr_deg = np.mean(np.linalg.norm(opt["theta"], axis=1)) * 180.0 / np.pi

        print(f"cost                  : {opt['cost0']:.3f} -> {opt['cost1']:.3f}")
        print(f"scale raw init/clipped: {opt['scale_init_raw']:.6f} -> {opt['scale_init_clipped']:.6f}")
        print(f"optimized scale       : {opt['scale']:.6f}")
        print(f"optimized accel bias  : {opt['ba']}")
        print(f"mean rot correction   : {mean_rot_corr_deg:.4f} deg")
        print(f"RAW VIO SE3 ATE       : {raw_res['ate']:.6f} m")
        print(f"RAW VIO SE3 RPE trans : {raw_res['rpe_t']:.6f} m")
        print(f"FILT VIO SE3 ATE      : {filtered_res['ate']:.6f} m")
        print(f"FILT VIO SE3 RPE trans: {filtered_res['rpe_t']:.6f} m")
        print(f"FILT VIO diag scale   : {diag_res['scale']:.6f}")
        print(f"FILT VIO diag ATE     : {diag_res['ate']:.6f} m")
        print(f"RAW ATE improvement   : {raw_ate_imp:.2f}%")
        print(f"RAW RPE improvement   : {raw_rpe_imp:.2f}%")
        print(f"FILT ATE improvement  : {filt_ate_imp:.2f}%")
        print(f"FILT RPE improvement  : {filt_rpe_imp:.2f}%")

        results.append({
            "name": name,
            "gravity": gravity,
            "opt": opt,
            "dense_raw": dense_raw,
            "dense_filtered": dense_filtered,
            "dense_R": dense_R,
            "raw_res": raw_res,
            "filtered_res": filtered_res,
            "diag_res": diag_res,
            "raw_ate_imp": raw_ate_imp,
            "raw_rpe_imp": raw_rpe_imp,
            "filt_ate_imp": filt_ate_imp,
            "filt_rpe_imp": filt_rpe_imp,
            "mean_rot_corr_deg": mean_rot_corr_deg,
        })

    if len(results) == 0:
        raise RuntimeError("All gravity tests failed.")

    best = min(results, key=lambda r: r["filtered_res"]["ate"])

    save_tum(OUT_TRAJ_RAW, vo_ts, best["dense_raw"], best["dense_R"])
    save_tum(OUT_TRAJ_FILTERED, vo_ts, best["dense_filtered"], best["dense_R"])
    plot_results(best["filtered_res"])

    print("\n==============================")
    print("FINAL WEEK 8-10 ROTATION-REFINED VIO VALUES")
    print("==============================")
    print(f"Best gravity mode          : {best['name']}")
    print(f"Gravity vector             : {best['gravity']}")
    print(f"Gravity norm               : {np.linalg.norm(best['gravity']):.6f}")
    print(f"Filter used                : {USE_FILTER}")
    print(f"Filter type                : {'Savitzky-Golay' if HAS_SAVGOL else 'Moving average'}")
    print(f"Filter window              : {FILTER_WINDOW}")
    print(f"Scale bounds               : [{SCALE_LOW:.3f}, {SCALE_HIGH:.3f}]")
    print(f"Mean rotation correction   : {best['mean_rot_corr_deg']:.4f} deg")
    print()
    print(f"VO associated poses        : {vo_res['associated']}")
    print(f"VO Sim(3) scale            : {vo_res['scale']:.6f}")
    print(f"VO ATE RMSE                : {vo_res['ate']:.6f} m")
    print(f"VO RPE trans RMSE          : {vo_res['rpe_t']:.6f} m")
    print(f"VO RPE rot RMSE            : {vo_res['rpe_r']:.6f} deg")
    print()
    print(f"RAW VIO ATE RMSE           : {best['raw_res']['ate']:.6f} m")
    print(f"RAW VIO RPE trans RMSE     : {best['raw_res']['rpe_t']:.6f} m")
    print(f"RAW VIO RPE rot RMSE       : {best['raw_res']['rpe_r']:.6f} deg")
    print()
    print(f"FILTERED VIO ATE RMSE      : {best['filtered_res']['ate']:.6f} m")
    print(f"FILTERED VIO RPE trans RMSE: {best['filtered_res']['rpe_t']:.6f} m")
    print(f"FILTERED VIO RPE rot RMSE  : {best['filtered_res']['rpe_r']:.6f} deg")
    print()
    print(f"Filtered VIO Sim(3) diag scale: {best['diag_res']['scale']:.6f}")
    print(f"Filtered VIO Sim(3) diag ATE  : {best['diag_res']['ate']:.6f} m")
    print()
    print(f"RAW VIO ATE improvement       : {best['raw_ate_imp']:.2f}%")
    print(f"RAW VIO RPE improvement       : {best['raw_rpe_imp']:.2f}%")
    print(f"FILTERED VIO ATE improvement  : {best['filt_ate_imp']:.2f}%")
    print(f"FILTERED VIO RPE improvement  : {best['filt_rpe_imp']:.2f}%")

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("WEEK 8-10 POSE-GRAPH VIO WITH ROTATION REFINEMENT\n")
        f.write("================================================\n\n")
        f.write(f"segment: {START_FRAME_INDEX}:{end_idx}\n")
        f.write(f"keyframe_step: {KEYFRAME_STEP}\n")
        f.write(f"last_keyframe_index: {kf_indices[-1]}\n")
        f.write(f"scale_bounds: [{SCALE_LOW}, {SCALE_HIGH}]\n")
        f.write(f"sigma_end_anchor: {SIGMA_END_ANCHOR}\n")
        f.write(f"sigma_ba_prior: {SIGMA_BA_PRIOR}\n")
        f.write(f"sigma_imu_rot: {SIGMA_IMU_ROT}\n")
        f.write(f"sigma_rot_prior: {SIGMA_ROT_PRIOR}\n")
        f.write(f"max_rot_corr: {MAX_ROT_CORR}\n")
        f.write(f"filter_used: {USE_FILTER}\n")
        f.write(f"filter_type: {'Savitzky-Golay' if HAS_SAVGOL else 'Moving average'}\n")
        f.write(f"filter_window: {FILTER_WINDOW}\n")
        f.write(f"filter_polyorder: {FILTER_POLYORDER}\n\n")
        f.write(f"vo_ate: {vo_res['ate']:.9f}\n")
        f.write(f"vo_rpe_trans: {vo_res['rpe_t']:.9f}\n")
        f.write(f"vo_rpe_rot_deg: {vo_res['rpe_r']:.9f}\n")
        f.write(f"best_gravity: {best['name']}\n")
        f.write(f"best_gravity_vector: {best['gravity'].tolist()}\n")
        f.write(f"optimized_scale: {best['opt']['scale']:.9f}\n")
        f.write(f"optimized_accel_bias: {best['opt']['ba'].tolist()}\n")
        f.write(f"mean_rotation_correction_deg: {best['mean_rot_corr_deg']:.9f}\n")
        f.write(f"raw_vio_ate: {best['raw_res']['ate']:.9f}\n")
        f.write(f"raw_vio_rpe_trans: {best['raw_res']['rpe_t']:.9f}\n")
        f.write(f"raw_vio_rpe_rot_deg: {best['raw_res']['rpe_r']:.9f}\n")
        f.write(f"filtered_vio_ate: {best['filtered_res']['ate']:.9f}\n")
        f.write(f"filtered_vio_rpe_trans: {best['filtered_res']['rpe_t']:.9f}\n")
        f.write(f"filtered_vio_rpe_rot_deg: {best['filtered_res']['rpe_r']:.9f}\n")
        f.write(f"filtered_diag_scale: {best['diag_res']['scale']:.9f}\n")
        f.write(f"filtered_diag_ate: {best['diag_res']['ate']:.9f}\n")
        f.write(f"raw_ate_improvement_percent: {best['raw_ate_imp']:.9f}\n")
        f.write(f"raw_rpe_improvement_percent: {best['raw_rpe_imp']:.9f}\n")
        f.write(f"filtered_ate_improvement_percent: {best['filt_ate_imp']:.9f}\n")
        f.write(f"filtered_rpe_improvement_percent: {best['filt_rpe_imp']:.9f}\n")
        f.write(f"raw_trajectory: {OUT_TRAJ_RAW}\n")
        f.write(f"filtered_trajectory: {OUT_TRAJ_FILTERED}\n")

    print("\nSaved:")
    print(OUT_TRAJ_RAW)
    print(OUT_TRAJ_FILTERED)
    print(OUT_METRICS)
    print(OUT_XY)
    print(OUT_XZ)
    print(OUT_ATE)


if __name__ == "__main__":
    run()
