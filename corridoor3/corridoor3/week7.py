import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\Admin\vio_project_shanza")

SEQ = ROOT / "data" / "corridor3_512_16" / "dataset-corridor3_512_16"

EST_FILE = ROOT / "result" / "week5_6_strong_reset_debug_corridor3" / "trajectory_tum.txt"
GT_FILE = SEQ / "mav0" / "mocap0" / "data.csv"
STATS_FILE = ROOT / "result" / "week5_6_strong_reset_debug_corridor3" / "stats_debug.csv"

OUT = ROOT / "result" / "week7_vo_eval_corridor3"
OUT.mkdir(parents=True, exist_ok=True)

OUT_XZ = OUT / "corridor3_vo_xz.png"
OUT_XY = OUT / "corridor3_vo_xy.png"
OUT_METRICS = OUT / "corridor3_week7_summary.txt"


# ============================================================
# LOADERS
# ============================================================

def load_est_tum(path):
    ts = []
    p = []

    if not path.exists():
        raise FileNotFoundError(f"Missing estimated trajectory:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            a = line.split()
            if len(a) < 4:
                continue

            ts.append(float(a[0]))
            p.append([float(a[1]), float(a[2]), float(a[3])])

    return np.asarray(ts), np.asarray(p)


def load_gt_positions(path):
    ts = []
    p = []

    if not path.exists():
        print("GT file not found. Start/end drift will be disabled.")
        return np.asarray(ts), np.asarray(p)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) < 4:
                continue

            ts.append(float(row[0]) * 1e-9)
            p.append([float(row[1]), float(row[2]), float(row[3])])

    return np.asarray(ts), np.asarray(p)


def load_stats(path):
    if not path.exists():
        return None

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = next(f, None)

        for line in f:
            a = line.strip().split(",")
            if len(a) < 9:
                continue
            rows.append(a)

    return rows


# ============================================================
# BASIC METRICS
# ============================================================

def path_length(p):
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def nearest_pose(ts, p, target_t):
    idx = int(np.argmin(np.abs(ts - target_t)))
    return ts[idx], p[idx]


def start_end_drift(est_ts, est_p, gt_ts, gt_p):
    if len(gt_ts) < 2 or len(est_ts) < 2:
        return None

    est_start_t, est_start_p = nearest_pose(est_ts, est_p, gt_ts[0])
    est_end_t, est_end_p = nearest_pose(est_ts, est_p, gt_ts[-1])

    est_delta = est_end_p - est_start_p
    gt_delta = gt_p[-1] - gt_p[0]

    # Monocular VO has arbitrary scale, so compare direction/shape carefully.
    scale = np.linalg.norm(gt_delta) / (np.linalg.norm(est_delta) + 1e-12)

    drift_vec = scale * est_delta - gt_delta
    drift = float(np.linalg.norm(drift_vec))

    gt_start_end = float(np.linalg.norm(gt_delta))
    drift_percent = 100.0 * drift / (gt_start_end + 1e-12)

    return {
        "scale_used_for_endpoint": scale,
        "drift_m": drift,
        "drift_percent": drift_percent,
        "gt_start_end_m": gt_start_end,
        "est_start_dt": abs(est_start_t - gt_ts[0]),
        "est_end_dt": abs(est_end_t - gt_ts[-1]),
    }


def stats_summary(rows):
    if rows is None:
        return None

    total = len(rows)
    statuses = {}

    for r in rows:
        status = r[-1]
        statuses[status] = statuses.get(status, 0) + 1

    return {
        "total_rows": total,
        "statuses": statuses,
    }


# ============================================================
# PLOTS
# ============================================================

def plot_raw_trajectory(est_p, gt_p=None):
    est = est_p - est_p[0]

    plt.figure(figsize=(8, 8))
    plt.plot(est[:, 0], est[:, 2], label="VO raw trajectory", linewidth=1.5)

    if gt_p is not None and len(gt_p) > 1:
        gt = gt_p - gt_p[0]
        plt.scatter([gt[0, 0], gt[-1, 0]], [gt[0, 2], gt[-1, 2]], label="GT start/end", s=40)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Corridor3 Week 7 VO: XZ qualitative trajectory")
    plt.tight_layout()
    plt.savefig(str(OUT_XZ), dpi=250)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(est[:, 0], est[:, 1], label="VO raw trajectory", linewidth=1.5)

    if gt_p is not None and len(gt_p) > 1:
        gt = gt_p - gt_p[0]
        plt.scatter([gt[0, 0], gt[-1, 0]], [gt[0, 1], gt[-1, 1]], label="GT start/end", s=40)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Corridor3 Week 7 VO: XY qualitative trajectory")
    plt.tight_layout()
    plt.savefig(str(OUT_XY), dpi=250)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== WEEK 7 VO EVALUATION: CORRIDOR3 ===")
    print("Note: Corridor3 does not use full ATE/RPE evaluation here.")
    print("Evaluation focuses on qualitative trajectory, start-end drift, and tracking stability.")

    est_ts, est_p = load_est_tum(EST_FILE)
    gt_ts, gt_p = load_gt_positions(GT_FILE)
    rows = load_stats(STATS_FILE)

    print("\nFiles:")
    print("EST_FILE  :", EST_FILE)
    print("GT_FILE   :", GT_FILE)
    print("STATS_FILE:", STATS_FILE)

    print("\nLoaded:")
    print("Estimated poses:", len(est_p))
    print("GT poses:", len(gt_p))
    print("Estimated path length raw:", f"{path_length(est_p):.3f}")

    drift = start_end_drift(est_ts, est_p, gt_ts, gt_p)
    stats = stats_summary(rows)

    print("\n=== START-END DRIFT ===")
    if drift is not None:
        print(f"GT start-end length      : {drift['gt_start_end_m']:.6f} m")
        print(f"Endpoint scale used      : {drift['scale_used_for_endpoint']:.9f}")
        print(f"Start-end drift          : {drift['drift_m']:.6f} m")
        print(f"Start-end drift percent  : {drift['drift_percent']:.3f} %")
        print(f"Start/end time mismatch  : {drift['est_start_dt']:.4f}s / {drift['est_end_dt']:.4f}s")
    else:
        print("Not available.")

    print("\n=== TRACKING / STATUS SUMMARY ===")
    if stats is not None:
        print("Total stats rows:", stats["total_rows"])
        for k, v in sorted(stats["statuses"].items()):
            print(f"{k}: {v}")
    else:
        print("Stats file not found.")

    plot_raw_trajectory(est_p, gt_p)

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("WEEK 7 VO EVALUATION: CORRIDOR3\n")
        f.write("================================\n")
        f.write("No full ATE/RPE is reported because Corridor3 has only limited/start-end GT.\n\n")

        f.write(f"Estimated poses: {len(est_p)}\n")
        f.write(f"GT poses: {len(gt_p)}\n")
        f.write(f"Estimated raw path length: {path_length(est_p):.6f}\n\n")

        if drift is not None:
            f.write("START-END DRIFT\n")
            f.write(f"GT start-end length: {drift['gt_start_end_m']:.9f} m\n")
            f.write(f"Endpoint scale used: {drift['scale_used_for_endpoint']:.9f}\n")
            f.write(f"Start-end drift: {drift['drift_m']:.9f} m\n")
            f.write(f"Start-end drift percent: {drift['drift_percent']:.9f} %\n")
            f.write(f"Start time mismatch: {drift['est_start_dt']:.9f} s\n")
            f.write(f"End time mismatch: {drift['est_end_dt']:.9f} s\n\n")

        if stats is not None:
            f.write("TRACKING STATUS SUMMARY\n")
            f.write(f"Total rows: {stats['total_rows']}\n")
            for k, v in sorted(stats["statuses"].items()):
                f.write(f"{k}: {v}\n")

        f.write("\nInterpretation:\n")
        f.write("Corridor3 is evaluated qualitatively using trajectory plausibility, drift behavior, tracking stability, and reset/failure counts.\n")

    print("\nSaved:")
    print(OUT_METRICS)
    print(OUT_XZ)
    print(OUT_XY)
    print("\nDONE.")


if __name__ == "__main__":
    main()