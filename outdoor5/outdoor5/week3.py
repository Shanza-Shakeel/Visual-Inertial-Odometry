import csv
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(r"C:\Users\Admin\vio_project_shanza")

SEQ_NAME = "outdoors5_512_16"  # Change this to the actual sequence name you want to load
DATASET_NAME = "dataset-outdoors5_512_16"

SEQ = ROOT / "data" / SEQ_NAME / DATASET_NAME
CAM0 = SEQ / "mav0" / "cam0"
GT0 = SEQ / "mav0" / "mocap0"

IMG_DIR = CAM0 / "data"
IMG_CSV = CAM0 / "data.csv"
GT_CSV = GT0 / "data.csv"
CALIB = SEQ / "dso" / "camchain.yaml"

OUT = ROOT / "result" / f"week3_4_outdoors5_low_reject"
OUT.mkdir(parents=True, exist_ok=True)

OUT_TRAJ = OUT / "trajectory_vo.txt"
OUT_STATS = OUT / "stats.csv"
OUT_PLOT = OUT / "trajectory_plot.png"

# ================= PARAMETERS =================
MAX_FEATURES = 1800
MIN_TRACKS = 40
MIN_INLIERS = 25
MIN_RATIO = 0.35
MIN_PARALLAX = 0.0004

REDETECT_IF_LESS = 300
REDETECT_EVERY = 5

MIN_TRIANGULATED = 3

# For monocular VO frontend only.
# Translation scale is arbitrary at this stage.
STEP_SCALE = 1.0


# ================= YAML / CALIB =================
def read_yaml(path):
    text = path.read_text(encoding="utf-8")
    if text.startswith("%YAML:1.0"):
        text = text.replace("%YAML:1.0", "", 1).lstrip()
    return yaml.safe_load(text)


def validate_rotation(R):
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
        raise RuntimeError("Rotation matrix is not orthogonal")

    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-5):
        raise RuntimeError(f"Invalid rotation determinant: {det}")


def load_calibration():
    cam = read_yaml(CALIB)["cam0"]

    fx, fy, cx, cy = cam["intrinsics"]

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    D = np.array(
        cam.get("distortion_coeffs", [0, 0, 0, 0]),
        dtype=np.float64
    )

    T_cam_imu = np.array(cam["T_cam_imu"], dtype=np.float64)
    validate_rotation(T_cam_imu[:3, :3])

    return K, D, T_cam_imu


# ================= DATA =================
def load_images():
    paths, timestamps = [], []

    with open(IMG_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) < 2:
                continue

            img_path = IMG_DIR / row[1].strip()

            if img_path.exists():
                paths.append(str(img_path))
                timestamps.append(float(row[0]) * 1e-9)

    if len(paths) < 2:
        raise RuntimeError("Not enough images found.")

    return paths, timestamps


def load_ground_truth():
    gt = []

    if not GT_CSV.exists():
        print("Ground truth file not found.")
        return gt

    with open(GT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) < 4:
                continue

            gt.append({
                "timestamp": float(row[0]) * 1e-9,
                "t": np.array(row[1:4], dtype=np.float64)
            })

    return gt


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    if img.shape != (512, 512):
        raise RuntimeError(f"Unexpected image shape: {img.shape}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# ================= FEATURES =================
def detect_features(img, max_features=MAX_FEATURES):
    """
    Stronger than your ORB-grid for outdoors5.
    Shi-Tomasi corners are better for KLT tracking.
    """
    pts = cv2.goodFeaturesToTrack(
        img,
        maxCorners=max_features,
        qualityLevel=0.0005,
        minDistance=5,
        blockSize=7
    )

    if pts is not None and len(pts) >= 40:
        return pts.astype(np.float32)

    # Last fallback: dense grid.
    h, w = img.shape
    grid = []

    for y in range(15, h - 15, 15):
        for x in range(15, w - 15, 15):
            grid.append([x, y])

    return np.asarray(grid, dtype=np.float32).reshape(-1, 1, 2)


def track_klt(prev_img, curr_img, prev_pts):
    if prev_pts is None or len(prev_pts) == 0:
        return None, None

    curr_pts, st1, _ = cv2.calcOpticalFlowPyrLK(
        prev_img,
        curr_img,
        prev_pts,
        None,
        winSize=(31, 31),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)
    )

    if curr_pts is None or st1 is None:
        return None, None

    back_pts, st2, _ = cv2.calcOpticalFlowPyrLK(
        curr_img,
        prev_img,
        curr_pts,
        None,
        winSize=(31, 31),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)
    )

    if back_pts is None or st2 is None:
        return None, None

    p0 = prev_pts.reshape(-1, 2)
    p1 = curr_pts.reshape(-1, 2)
    pb = back_pts.reshape(-1, 2)

    fb_err = np.linalg.norm(p0 - pb, axis=1)

    h, w = curr_img.shape

    good = (
        (st1.ravel() == 1)
        & (st2.ravel() == 1)
        & (fb_err < 3.0)
        & np.isfinite(p1).all(axis=1)
        & (p1[:, 0] >= 3)
        & (p1[:, 0] < w - 3)
        & (p1[:, 1] >= 3)
        & (p1[:, 1] < h - 3)
    )

    return (
        prev_pts[good].reshape(-1, 1, 2),
        curr_pts[good].reshape(-1, 1, 2)
    )


# ================= GEOMETRY =================
def undistort_points_fisheye(pts, K, D):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)

    out = cv2.fisheye.undistortPoints(
        pts,
        K,
        D,
        R=np.eye(3),
        P=np.eye(3)
    )

    return out.reshape(-1, 2)


def estimate_motion(p1, p2, K, D):
    if p1 is None or p2 is None:
        return None, None, None, 0, 0.0, 0.0

    if len(p1) < MIN_TRACKS:
        return None, None, None, 0, 0.0, 0.0

    n1 = undistort_points_fisheye(p1, K, D)
    n2 = undistort_points_fisheye(p2, K, D)

    parallax = float(np.median(np.linalg.norm(n1 - n2, axis=1)))

    if parallax < MIN_PARALLAX:
        return None, None, None, 0, 0.0, parallax

    best = None

    # Adaptive thresholds for outdoors.
    for th in [0.003, 0.005, 0.008, 0.012]:
        E, mask = cv2.findEssentialMat(
            n1,
            n2,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=th
        )

        if E is None or mask is None:
            continue

        mask = mask.ravel().astype(bool)
        inliers = int(mask.sum())
        ratio = inliers / max(1, len(mask))

        if best is None or inliers > best["inliers"]:
            best = {
                "E": E,
                "mask": mask,
                "inliers": inliers,
                "ratio": ratio,
                "threshold": th
            }

        if inliers >= MIN_INLIERS and ratio >= MIN_RATIO:
            break

    if best is None:
        return None, None, None, 0, 0.0, parallax

    if best["inliers"] < MIN_INLIERS or best["ratio"] < MIN_RATIO:
        return None, None, best["mask"], best["inliers"], best["ratio"], parallax

    try:
        retval, R, t, _ = cv2.recoverPose(
            best["E"],
            n1[best["mask"]],
            n2[best["mask"]],
            focal=1.0,
            pp=(0.0, 0.0)
        )
    except cv2.error:
        return None, None, best["mask"], best["inliers"], best["ratio"], parallax

    if retval < 15:
        return None, None, best["mask"], best["inliers"], best["ratio"], parallax

    validate_rotation(R)

    t = t.reshape(3)
    t_norm = np.linalg.norm(t)

    if t_norm < 1e-12:
        return None, None, best["mask"], best["inliers"], best["ratio"], parallax

    t = t / t_norm

    return R, t, best["mask"], best["inliers"], best["ratio"], parallax


def triangulate_points(p1, p2, R, t, K, D):
    n1 = undistort_points_fisheye(p1, K, D)
    n2 = undistort_points_fisheye(p2, K, D)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])

    Xh = cv2.triangulatePoints(P1, P2, n1.T, n2.T)
    X = Xh[:3] / (Xh[3] + 1e-12)

    X2 = R @ X + t.reshape(3, 1)

    valid = (
        np.isfinite(X).all(axis=0)
        & (X[2] > 0)
        & (X2[2] > 0)
    )

    return X[:, valid].T


# ================= OUTPUT =================
def save_tracking_image(img, p1, p2, frame_id):
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    a = p1.reshape(-1, 2)
    b = p2.reshape(-1, 2)

    for x, y in zip(a, b):
        cv2.circle(canvas, (int(y[0]), int(y[1])), 2, (0, 255, 0), -1)
        cv2.line(
            canvas,
            (int(x[0]), int(x[1])),
            (int(y[0]), int(y[1])),
            (0, 180, 255),
            1
        )

    cv2.imwrite(str(OUT / f"tracking_{frame_id:05d}.png"), canvas)


def save_outputs(timestamps, trajectory, stats):
    with open(OUT_TRAJ, "w", encoding="utf-8") as f:
        for ts, p in zip(timestamps, trajectory):
            f.write(f"{ts:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write("frame,tracked,inliers,ratio,parallax,status\n")
        for row in stats:
            f.write("{},{},{},{:.6f},{:.6f},{}\n".format(*row))


def plot_trajectory(traj, gt):
    vo = np.asarray(traj)

    plt.figure(figsize=(8, 8))
    plt.plot(vo[:, 0], vo[:, 2], linewidth=1.2, label="VO frontend")

    if len(gt) > 0:
        gt_xyz = np.array([g["t"] for g in gt])
        gt_xyz = gt_xyz - gt_xyz[0]
        plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], linewidth=1.2, label="GT")

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Week 3-4 VO Frontend - {SEQ_NAME}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_PLOT), dpi=200)
    plt.close()


# ================= MAIN =================
def run():
    K, D, _ = load_calibration()
    paths, timestamps = load_images()
    gt = load_ground_truth()

    print("=== WEEK 3-4 OUTDOORS5 LOW-REJECTION VO FRONTEND ===")
    print("Frames:", len(paths))

    prev_img = read_img(paths[0])
    prev_pts = detect_features(prev_img)

    if prev_pts is None or len(prev_pts) < 80:
        raise RuntimeError("Initial feature detection failed.")

    T_wc = np.eye(4, dtype=np.float64)

    trajectory = [T_wc[:3, 3].copy()]
    stats = []

    ok_count = 0
    hold_count = 0
    reset_count = 0
    geom_fail = 0
    triang_fail = 0

    for i in range(1, len(paths)):
        curr_img = read_img(paths[i])

        tracked_prev, tracked_curr = track_klt(prev_img, curr_img, prev_pts)

        if tracked_curr is None or len(tracked_curr) < MIN_TRACKS:
            reset_count += 1
            hold_count += 1

            new_pts = detect_features(curr_img)

            if new_pts is not None and len(new_pts) >= 40:
                prev_pts = new_pts
                prev_img = curr_img.copy()

            trajectory.append(T_wc[:3, 3].copy())
            stats.append((i, 0, 0, 0.0, 0.0, "HOLD_TRACK_RESET"))
            continue

        R, t, mask, inliers, ratio, parallax = estimate_motion(
            tracked_prev,
            tracked_curr,
            K,
            D
        )

        if R is None or t is None:
            geom_fail += 1
            hold_count += 1

            prev_img = curr_img.copy()
            prev_pts = tracked_curr.copy()

            if len(prev_pts) < REDETECT_IF_LESS or i % REDETECT_EVERY == 0:
                new_pts = detect_features(curr_img)
                if new_pts is not None and len(new_pts) >= 40:
                    prev_pts = new_pts

            trajectory.append(T_wc[:3, 3].copy())
            stats.append((i, len(tracked_curr), inliers, ratio, parallax, "HOLD_GEOM"))
            continue

        pts3d = triangulate_points(
            tracked_prev[mask],
            tracked_curr[mask],
            R,
            t,
            K,
            D
        )

        # Do not reject good pose just because triangulation is weak.
        # But record it.
        status = "OK"
        if len(pts3d) < MIN_TRIANGULATED:
            triang_fail += 1
            status = "OK_WEAK_TRIANG"

        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t * STEP_SCALE

        T_wc = T_wc @ np.linalg.inv(T_rel)

        trajectory.append(T_wc[:3, 3].copy())
        ok_count += 1

        stats.append((i, len(tracked_curr), inliers, ratio, parallax, status))

        if i % 100 == 0:
            print(
                f"frame={i:05d} "
                f"OK={ok_count} HOLD={hold_count} "
                f"inliers={inliers} ratio={ratio:.3f} "
                f"parallax={parallax:.4f} triangulated={len(pts3d)}"
            )
            save_tracking_image(curr_img, tracked_prev, tracked_curr, i)

        prev_img = curr_img.copy()
        prev_pts = tracked_curr.copy()

        if len(prev_pts) < REDETECT_IF_LESS or i % REDETECT_EVERY == 0:
            new_pts = detect_features(curr_img)
            if new_pts is not None and len(new_pts) >= 40:
                prev_pts = new_pts

    save_outputs(timestamps, trajectory, stats)
    plot_trajectory(trajectory, gt)

    print("\nDONE")
    print(f"Input frames: {len(paths)}")
    print(f"Trajectory poses saved: {len(trajectory)}")
    print(f"OK frames: {ok_count}")
    print(f"Held frames: {hold_count}")
    print(f"Resets: {reset_count}")
    print(f"Geometry fails: {geom_fail}")
    print(f"Weak triangulation frames accepted: {triang_fail}")
    print("Trajectory:", OUT_TRAJ)
    print("Stats:", OUT_STATS)
    print("Plot:", OUT_PLOT)


if __name__ == "__main__":
    run()