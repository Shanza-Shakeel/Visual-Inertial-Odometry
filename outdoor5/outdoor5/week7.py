import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\Admin\vio_project_shanza")
SEQ = ROOT / "data" / "outdoors5_512_16" / "dataset-outdoors5_512_16"

EST_FILE = ROOT / "result" / "week5_6_strong_reset_debug_outdoors5" / "trajectory_tum.txt"
GT_FILE = SEQ / "mav0" / "mocap0" / "data.csv"

OUT = ROOT / "result" / "week7_vo_eval_outdoors5"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ASSOC_DT = 0.02
RPE_DELTA = 20


def load_est(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            v = line.split()
            ts = float(v[0])
            p = np.array(list(map(float, v[1:4])), dtype=np.float64)
            data.append((ts, p))
    return data


def load_gt(path):
    data = []
    with open(path, "r") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ts = float(row[0]) * 1e-9
            p = np.array(list(map(float, row[1:4])), dtype=np.float64)
            data.append((ts, p))
    return data


def associate(est, gt, max_dt=MAX_ASSOC_DT):
    gt_ts = np.array([x[0] for x in gt])
    pairs = []

    j = 0
    for ts, p in est:
        while j + 1 < len(gt_ts) and abs(gt_ts[j + 1] - ts) < abs(gt_ts[j] - ts):
            j += 1

        if abs(gt_ts[j] - ts) <= max_dt:
            pairs.append((p, gt[j][1], ts))

    if len(pairs) < 20:
        raise RuntimeError(f"Too few associated poses: {len(pairs)}")

    est_p = np.array([x[0] for x in pairs])
    gt_p = np.array([x[1] for x in pairs])
    ts = np.array([x[2] for x in pairs])

    return ts, est_p, gt_p


def sim3_umeyama(src, dst):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c / len(src)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var = np.mean(np.sum(src_c ** 2, axis=1))
    scale = np.sum(S) / (var + 1e-12)

    t = dst_mean - scale * R @ src_mean
    aligned = scale * (R @ src.T).T + t

    return aligned, scale, R, t


def ate_rmse(aligned, gt):
    err = np.linalg.norm(aligned - gt, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), err


def rpe_translation(aligned, gt, delta=RPE_DELTA):
    errs = []
    for i in range(len(aligned) - delta):
        de = aligned[i + delta] - aligned[i]
        dg = gt[i + delta] - gt[i]
        errs.append(np.linalg.norm(de - dg))

    errs = np.array(errs)
    return float(np.sqrt(np.mean(errs ** 2))), errs


def path_length(x):
    return float(np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1)))


def save_aligned_trajectory(ts, aligned):
    out_file = OUT / "vo_aligned_sim3.txt"
    with open(out_file, "w") as f:
        for t, p in zip(ts, aligned):
            f.write(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return out_file


def main():
    est = load_est(EST_FILE)
    gt = load_gt(GT_FILE)

    ts, est_p, gt_p = associate(est, gt)

    # Do NOT subtract first pose before Sim(3). Sim(3) already estimates translation.
    aligned, scale, R, t = sim3_umeyama(est_p, gt_p)

    ate, ate_err = ate_rmse(aligned, gt_p)
    rpe, rpe_err = rpe_translation(aligned, gt_p)

    drift = np.linalg.norm((aligned[-1] - aligned[0]) - (gt_p[-1] - gt_p[0]))
    gt_len = path_length(gt_p)
    gt_start_end = np.linalg.norm(gt_p[-1] - gt_p[0])

    drift_pct_path = 100.0 * drift / gt_len
    drift_pct_start_end = 100.0 * drift / gt_start_end if gt_start_end > 1e-12 else np.nan

    aligned_file = save_aligned_trajectory(ts, aligned)

    print("\n=== WEEK 7 VO EVALUATION - OUTDOORS5 ===")
    print("Estimated poses:", len(est))
    print("GT poses:", len(gt))
    print("Associated poses:", len(ts))
    print(f"Sim(3) scale: {scale:.6f}")
    print(f"ATE RMSE: {ate:.4f} m")
    print(f"RPE RMSE delta={RPE_DELTA}: {rpe:.4f} m")
    print(f"Start-end drift: {drift:.4f} m")
    print(f"GT path length: {gt_len:.4f} m")
    print(f"Drift percentage by path length: {drift_pct_path:.2f} %")
    print(f"Drift percentage by start-end displacement: {drift_pct_start_end:.2f} %")

    with open(OUT / "metrics.txt", "w") as f:
        f.write("=== WEEK 7 VO EVALUATION - OUTDOORS5 ===\n")
        f.write(f"Estimated poses: {len(est)}\n")
        f.write(f"GT poses: {len(gt)}\n")
        f.write(f"Associated poses: {len(ts)}\n")
        f.write(f"Sim(3) scale: {scale:.6f}\n")
        f.write(f"ATE RMSE: {ate:.4f} m\n")
        f.write(f"RPE RMSE delta={RPE_DELTA}: {rpe:.4f} m\n")
        f.write(f"Start-end drift: {drift:.4f} m\n")
        f.write(f"GT path length: {gt_len:.4f} m\n")
        f.write(f"Drift percentage by path length: {drift_pct_path:.2f} %\n")
        f.write(f"Drift percentage by start-end displacement: {drift_pct_start_end:.2f} %\n")

    plt.figure(figsize=(7, 7))
    plt.plot(gt_p[:, 0], gt_p[:, 1], label="GT")
    plt.plot(aligned[:, 0], aligned[:, 1], label="VO aligned")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Outdoors5 VO vs GT - XY")
    plt.savefig(OUT / "trajectory_xy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(gt_p[:, 0], gt_p[:, 2], label="GT")
    plt.plot(aligned[:, 0], aligned[:, 2], label="VO aligned")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Outdoors5 VO vs GT - XZ")
    plt.savefig(OUT / "trajectory_xz.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ate_err)
    plt.grid(True)
    plt.xlabel("Associated pose index")
    plt.ylabel("ATE error [m]")
    plt.title("ATE over trajectory")
    plt.savefig(OUT / "ate_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(rpe_err)
    plt.grid(True)
    plt.xlabel("Pose index")
    plt.ylabel("RPE translation error [m]")
    plt.title(f"RPE translation, delta={RPE_DELTA}")
    plt.savefig(OUT / "rpe_error.png", dpi=200)
    plt.close()

    print("\nSaved to:", OUT)
    print("Aligned trajectory:", aligned_file)


if __name__ == "__main__":
    main()