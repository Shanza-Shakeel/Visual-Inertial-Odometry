import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\Shanza\Desktop\Semester 2\Visual-Inertial-Odometry")
SEQ = ROOT / "data" / "room2" / "dataset-room2_512_16"

GT_FILE = SEQ / "mav0" / "mocap0" / "data.csv"

VO_FILE = ROOT / "result" / "week5_6_strong_reset_debug" / "trajectory_tum.txt"

VIO_FILE = ROOT / "result" / "week8_10_posegraph_vio_rotation_refined" / "posegraph_vio_filtered_tum.txt"

OUT = ROOT / "result" / "week11_final_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

OUT_METRICS = OUT / "week11_metrics.csv"
OUT_TRAJ_XY = OUT / "trajectory_xy_vo_vio_gt.png"
OUT_TRAJ_XZ = OUT / "trajectory_xz_vo_vio_gt.png"
OUT_ATE = OUT / "ate_over_time.png"
OUT_RPE = OUT / "rpe_translation.png"


# ============================================================
# CONFIG
# ============================================================

MAX_ASSOC_DT = 0.02
RPE_DELTA = 20


# ============================================================
# ROTATION HELPERS
# ============================================================

def quat_to_R(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12

    x, y, z, w = q

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def log_so3(R):
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(c)

    if th < 1e-12:
        return np.zeros(3)

    W = (R - R.T) / (2.0 * np.sin(th))
    return th * np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


# ============================================================
# LOADERS
# ============================================================

def load_tum(path):
    ts, p, R = [], [], []

    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file:\n{path}")

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

            # mocap format: timestamp, px, py, pz, qw, qx, qy, qz
            qw = float(row[4])
            qx = float(row[5])
            qy = float(row[6])
            qz = float(row[7])

            R.append(quat_to_R(qx, qy, qz, qw))

    return np.asarray(ts), np.asarray(p), np.asarray(R)


# ============================================================
# ASSOCIATION
# ============================================================

def associate(est_ts, est_p, est_R, gt_ts, gt_p, gt_R):
    out_ts, ep, eR, gp, gR = [], [], [], [], []

    j = 0

    for i, t in enumerate(est_ts):
        while j + 1 < len(gt_ts) and abs(gt_ts[j + 1] - t) < abs(gt_ts[j] - t):
            j += 1

        if abs(gt_ts[j] - t) <= MAX_ASSOC_DT:
            out_ts.append(t)
            ep.append(est_p[i])
            eR.append(est_R[i])
            gp.append(gt_p[j])
            gR.append(gt_R[j])

    return (
        np.asarray(out_ts),
        np.asarray(ep),
        np.asarray(eR),
        np.asarray(gp),
        np.asarray(gR),
    )


# ============================================================
# ALIGNMENT
# ============================================================

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
    scale = np.trace(np.diag(S) @ D) / (var + 1e-12)

    t = mu_d - scale * R @ mu_s

    aligned = (scale * (R @ src.T)).T + t

    return aligned, R, t, scale


def umeyama_se3(src, dst):
    src = np.asarray(src)
    dst = np.asarray(dst)

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

    aligned = (R @ src.T).T + t

    return aligned, R, t, 1.0


# ============================================================
# METRICS
# ============================================================

def ate_rmse(est, gt):
    err = np.linalg.norm(est - gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "max": float(np.max(err)),
        "series": err,
    }


def rpe(est_p, est_R, gt_p, gt_R, delta=RPE_DELTA):
    trans_err = []
    rot_err = []

    for i in range(len(est_p) - delta):
        est_rel_t = est_p[i + delta] - est_p[i]
        gt_rel_t = gt_p[i + delta] - gt_p[i]
        trans_err.append(np.linalg.norm(est_rel_t - gt_rel_t))

        est_rel_R = est_R[i].T @ est_R[i + delta]
        gt_rel_R = gt_R[i].T @ gt_R[i + delta]
        rot_err.append(np.linalg.norm(log_so3(gt_rel_R.T @ est_rel_R)) * 180.0 / np.pi)

    trans_err = np.asarray(trans_err)
    rot_err = np.asarray(rot_err)

    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_err ** 2))),
        "trans_mean": float(np.mean(trans_err)),
        "trans_median": float(np.median(trans_err)),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot_err ** 2))),
        "rot_mean_deg": float(np.mean(rot_err)),
        "rot_median_deg": float(np.median(rot_err)),
        "trans_series": trans_err,
        "rot_series": rot_err,
    }


def start_end_drift(est, gt):
    est_rel = est[-1] - est[0]
    gt_rel = gt[-1] - gt[0]

    drift_abs = np.linalg.norm(est_rel - gt_rel)
    gt_dist = np.linalg.norm(gt_rel)

    drift_percent = 100.0 * drift_abs / gt_dist if gt_dist > 1e-12 else np.nan

    return float(drift_abs), float(drift_percent)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(name, est_ts, est_p, est_R, gt_ts, gt_p, gt_R, alignment):
    ts, ep, eR, gp, gR = associate(est_ts, est_p, est_R, gt_ts, gt_p, gt_R)

    if len(ep) < 50:
        raise RuntimeError(f"Too few associated poses for {name}: {len(ep)}")

    if alignment == "sim3":
        p_aligned, R_align, t_align, scale = umeyama_sim3(ep, gp)
    elif alignment == "se3":
        p_aligned, R_align, t_align, scale = umeyama_se3(ep, gp)
    else:
        raise ValueError("alignment must be 'sim3' or 'se3'")

    R_aligned = np.asarray([R_align @ Ri for Ri in eR])

    ate = ate_rmse(p_aligned, gp)
    rpe_res = rpe(p_aligned, R_aligned, gp, gR)
    drift_abs, drift_percent = start_end_drift(p_aligned, gp)

    return {
        "name": name,
        "alignment": alignment,
        "associated": len(ep),
        "scale": scale,
        "ts": ts,
        "p": p_aligned,
        "R": R_aligned,
        "gt_p": gp,
        "gt_R": gR,
        "ate": ate,
        "rpe": rpe_res,
        "drift_abs": drift_abs,
        "drift_percent": drift_percent,
    }


# ============================================================
# PLOTS
# ============================================================

def plot_xy(vo, vio):
    gt = vio["gt_p"]

    n = min(len(gt), len(vo["p"]), len(vio["p"]))

    plt.figure(figsize=(8, 8))
    plt.plot(gt[:n, 0], gt[:n, 1], label="GT", linewidth=2)
    plt.plot(vo["p"][:n, 0], vo["p"][:n, 1], label="VO Sim(3)", linewidth=2)
    plt.plot(vio["p"][:n, 0], vio["p"][:n, 1], label="VIO SE(3)", linewidth=2)
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Week 11: GT vs VO vs VIO XY")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_TRAJ_XY), dpi=250)
    plt.close()


def plot_xz(vo, vio):
    gt = vio["gt_p"]

    n = min(len(gt), len(vo["p"]), len(vio["p"]))

    plt.figure(figsize=(8, 8))
    plt.plot(gt[:n, 0], gt[:n, 2], label="GT", linewidth=2)
    plt.plot(vo["p"][:n, 0], vo["p"][:n, 2], label="VO Sim(3)", linewidth=2)
    plt.plot(vio["p"][:n, 0], vio["p"][:n, 2], label="VIO SE(3)", linewidth=2)
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Week 11: GT vs VO vs VIO XZ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_TRAJ_XZ), dpi=250)
    plt.close()


def plot_ate(vo, vio):
    n = min(len(vo["ate"]["series"]), len(vio["ate"]["series"]))

    plt.figure(figsize=(10, 4))
    plt.plot(vo["ts"][:n] - vo["ts"][0], vo["ate"]["series"][:n], label="VO ATE")
    plt.plot(vio["ts"][:n] - vio["ts"][0], vio["ate"]["series"][:n], label="VIO ATE")
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("ATE [m]")
    plt.title("Week 11: ATE over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_ATE), dpi=250)
    plt.close()


def plot_rpe(vo, vio):
    n = min(len(vo["rpe"]["trans_series"]), len(vio["rpe"]["trans_series"]))

    plt.figure(figsize=(10, 4))
    plt.plot(vo["rpe"]["trans_series"][:n], label="VO RPE trans")
    plt.plot(vio["rpe"]["trans_series"][:n], label="VIO RPE trans")
    plt.grid(True)
    plt.xlabel("RPE index")
    plt.ylabel("RPE translation [m]")
    plt.title("Week 11: RPE translation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_RPE), dpi=250)
    plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(vo, vio):
    rows = []

    for r in [vo, vio]:
        rows.append({
            "method": r["name"],
            "alignment": r["alignment"],
            "associated_poses": r["associated"],
            "scale": r["scale"],
            "ATE_RMSE_m": r["ate"]["rmse"],
            "ATE_mean_m": r["ate"]["mean"],
            "ATE_median_m": r["ate"]["median"],
            "ATE_max_m": r["ate"]["max"],
            "RPE_trans_RMSE_m": r["rpe"]["trans_rmse"],
            "RPE_trans_mean_m": r["rpe"]["trans_mean"],
            "RPE_trans_median_m": r["rpe"]["trans_median"],
            "RPE_rot_RMSE_deg": r["rpe"]["rot_rmse_deg"],
            "RPE_rot_mean_deg": r["rpe"]["rot_mean_deg"],
            "RPE_rot_median_deg": r["rpe"]["rot_median_deg"],
            "start_end_drift_m": r["drift_abs"],
            "start_end_drift_percent": r["drift_percent"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_METRICS, index=False)


def print_result(r):
    print(f"\n=== {r['name']} RESULT ===")
    print(f"Alignment              : {r['alignment']}")
    print(f"Associated poses       : {r['associated']}")
    print(f"Scale                  : {r['scale']:.6f}")
    print(f"ATE RMSE               : {r['ate']['rmse']:.6f} m")
    print(f"ATE mean               : {r['ate']['mean']:.6f} m")
    print(f"ATE median             : {r['ate']['median']:.6f} m")
    print(f"ATE max                : {r['ate']['max']:.6f} m")
    print(f"RPE trans RMSE         : {r['rpe']['trans_rmse']:.6f} m")
    print(f"RPE rot RMSE           : {r['rpe']['rot_rmse_deg']:.6f} deg")
    print(f"Start-end drift        : {r['drift_abs']:.6f} m")
    print(f"Start-end drift percent: {r['drift_percent']:.3f} %")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== WEEK 11 FINAL EVALUATION ===")

    gt_ts, gt_p, gt_R = load_gt(GT_FILE)
    vo_ts, vo_p, vo_R = load_tum(VO_FILE)
    vio_ts, vio_p, vio_R = load_tum(VIO_FILE)

    print(f"GT poses : {len(gt_ts)}")
    print(f"VO poses : {len(vo_ts)}")
    print(f"VIO poses: {len(vio_ts)}")

    # Monocular VO needs Sim(3)
    vo_res = evaluate_method(
        "VO",
        vo_ts,
        vo_p,
        vo_R,
        gt_ts,
        gt_p,
        gt_R,
        alignment="sim3"
    )

    # VIO is supposed to be metric, evaluate with SE(3)
    vio_res = evaluate_method(
        "VIO",
        vio_ts,
        vio_p,
        vio_R,
        gt_ts,
        gt_p,
        gt_R,
        alignment="se3"
    )

    print_result(vo_res)
    print_result(vio_res)

    save_metrics(vo_res, vio_res)
    plot_xy(vo_res, vio_res)
    plot_xz(vo_res, vio_res)
    plot_ate(vo_res, vio_res)
    plot_rpe(vo_res, vio_res)

    print("\nSaved:")
    print(OUT_METRICS)
    print(OUT_TRAJ_XY)
    print(OUT_TRAJ_XZ)
    print(OUT_ATE)
    print(OUT_RPE)

    print("\nDONE.")
    print("Use these final values honestly. If VIO improves RPE but not ATE, say exactly that.")


if __name__ == "__main__":
    main()