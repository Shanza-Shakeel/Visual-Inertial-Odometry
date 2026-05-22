import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ============================================================
# PATH CONFIG
# ============================================================

ROOT = Path(r"C:\Users\Shanza\Desktop\Semester 2\Visual-Inertial-Odometry")

SEQ = ROOT / "data" / "room2" / "dataset-room2_512_16"

CAM0 = SEQ / "mav0" / "cam0"
GT0 = SEQ / "mav0" / "mocap0"

IMG_DIR = CAM0 / "data"

IMG_CSV = CAM0 / "data.csv"
GT_CSV = GT0 / "data.csv"

CALIB = SEQ / "dso" / "camchain.yaml"

OUT = ROOT / "result" / "week3_4_improved"
OUT.mkdir(parents=True, exist_ok=True)

OUT_TRAJ = OUT / "trajectory_vo.txt"
OUT_STATS = OUT / "stats.csv"
OUT_PLOT = OUT / "trajectory_plot.png"


# ============================================================
# YAML LOADER
# ============================================================

def read_yaml(path):
    text = path.read_text(encoding="utf-8")

    if text.startswith("%YAML:1.0"):
        text = text.replace("%YAML:1.0", "", 1).lstrip()

    return yaml.safe_load(text)


# ============================================================
# CALIBRATION
# ============================================================

def validate_rotation(R):
    should_be_identity = R @ R.T

    if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
        raise RuntimeError("Rotation matrix is not orthogonal")

    det = np.linalg.det(R)

    if not np.isclose(det, 1.0, atol=1e-6):
        raise RuntimeError(f"Invalid rotation determinant: {det}")


def load_calibration():
    data = read_yaml(CALIB)

    cam = data["cam0"]

    fx, fy, cx, cy = cam["intrinsics"]

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )

    if "distortion_coeffs" in cam:
        D = np.array(cam["distortion_coeffs"], dtype=np.float64)
    elif "distortion_coefficients" in cam:
        D = np.array(cam["distortion_coefficients"], dtype=np.float64)
    else:
        D = np.zeros(4, dtype=np.float64)

    T_cam_imu = np.array(cam["T_cam_imu"], dtype=np.float64)

    if T_cam_imu.shape != (4, 4):
        raise RuntimeError("T_cam_imu must be 4x4")

    validate_rotation(T_cam_imu[:3, :3])

    return K, D, T_cam_imu


# ============================================================
# DATA LOADING
# ============================================================

def load_images():
    paths = []
    timestamps = []

    with open(IMG_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        next(reader)

        for row in reader:

            if len(row) < 2:
                continue

            ts = float(row[0]) * 1e-9

            img_path = IMG_DIR / row[1].strip()

            if img_path.exists():
                paths.append(str(img_path))
                timestamps.append(ts)

    if len(paths) < 2:
        raise RuntimeError("Not enough images found")

    return paths, timestamps


def load_ground_truth():
    gt = []

    if not GT_CSV.exists():
        print("Ground truth file not found")
        return gt

    with open(GT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        next(reader)

        for row in reader:

            ts = float(row[0]) * 1e-9

            tx, ty, tz = map(float, row[1:4])

            qx, qy, qz, qw = map(float, row[4:8])

            gt.append(
                {
                    "timestamp": ts,
                    "t": np.array([tx, ty, tz], dtype=np.float64),
                    "q": np.array([qx, qy, qz, qw], dtype=np.float64)
                }
            )

    return gt


# ============================================================
# IMAGE READING
# ============================================================

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    if img.shape != (512, 512):
        raise RuntimeError(f"Unexpected image shape: {img.shape}")

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(img)


# ============================================================
# FEATURE DETECTION
# ============================================================

def detect_orb_grid(
        img,
        max_features=700,
        grid_rows=4,
        grid_cols=4):

    h, w = img.shape

    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=7
    )

    pts = []

    cell_h = h // grid_rows
    cell_w = w // grid_cols

    per_cell = max_features // (grid_rows * grid_cols)

    for r in range(grid_rows):

        for c in range(grid_cols):

            y0 = r * cell_h
            y1 = h if r == grid_rows - 1 else (r + 1) * cell_h

            x0 = c * cell_w
            x1 = w if c == grid_cols - 1 else (c + 1) * cell_w

            cell = img[y0:y1, x0:x1]

            kps = orb.detect(cell, None)

            if not kps:
                continue

            kps = sorted(
                kps,
                key=lambda k: k.response,
                reverse=True
            )[:per_cell]

            for kp in kps:

                x = kp.pt[0] + x0
                y = kp.pt[1] + y0

                if 5 <= x < w - 5 and 5 <= y < h - 5:
                    pts.append([x, y])

    if len(pts) == 0:
        return None

    return np.asarray(
        pts,
        dtype=np.float32
    ).reshape(-1, 1, 2)


# ============================================================
# KLT TRACKING
# ============================================================

def track_klt(prev_img, curr_img, prev_pts):

    if prev_pts is None or len(prev_pts) == 0:
        return None, None

    next_pts, st1, _ = cv2.calcOpticalFlowPyrLK(
        prev_img,
        curr_img,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if next_pts is None or st1 is None:
        return None, None

    back_pts, st2, _ = cv2.calcOpticalFlowPyrLK(
        curr_img,
        prev_img,
        next_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if back_pts is None or st2 is None:
        return None, None

    fb_err = np.linalg.norm(
        prev_pts.reshape(-1, 2) -
        back_pts.reshape(-1, 2),
        axis=1
    )

    q = next_pts.reshape(-1, 2)

    h, w = curr_img.shape

    good = (
        (st1.ravel() == 1)
        & (st2.ravel() == 1)
        & (fb_err < 1.2)
        & np.isfinite(q).all(axis=1)
        & (q[:, 0] >= 5)
        & (q[:, 0] < w - 5)
        & (q[:, 1] >= 5)
        & (q[:, 1] < h - 5)
    )

    return (
        prev_pts[good].reshape(-1, 1, 2),
        next_pts[good].reshape(-1, 1, 2)
    )


# ============================================================
# FISHEYE UNDISTORTION
# ============================================================

def undistort_points_fisheye(pts, K, D):

    pts = pts.reshape(-1, 1, 2)

    undistorted = cv2.fisheye.undistortPoints(
        pts,
        K,
        D,
        R=np.eye(3),
        P=np.eye(3)
    )

    return undistorted.reshape(-1, 2)


# ============================================================
# MOTION ESTIMATION
# ============================================================

def estimate_motion(p1, p2, K, D):

    if p1 is None or p2 is None:
        return None, None, None, 0, 0.0, 0.0

    if len(p1) < 80:
        return None, None, None, 0, 0.0, 0.0

    pts1_n = undistort_points_fisheye(p1, K, D)
    pts2_n = undistort_points_fisheye(p2, K, D)

    parallax = np.median(
        np.linalg.norm(
            pts1_n - pts2_n,
            axis=1
        )
    )

    if parallax < 0.0015:
        return None, None, None, 0, 0.0, parallax

    E, mask = cv2.findEssentialMat(
        pts1_n,
        pts2_n,
        focal=1.0,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.003
    )

    if E is None or mask is None:
        return None, None, None, 0, 0.0, parallax

    mask = mask.ravel().astype(bool)

    inliers = int(mask.sum())

    ratio = inliers / max(1, len(mask))

    if inliers < 50 or ratio < 0.6:
        return None, None, mask, inliers, ratio, parallax

    retval, R, t, pose_mask = cv2.recoverPose(
        E,
        pts1_n[mask],
        pts2_n[mask],
        focal=1.0,
        pp=(0.0, 0.0)
    )

    if retval < 30:
        return None, None, mask, inliers, ratio, parallax

    validate_rotation(R)

    t_norm = np.linalg.norm(t)

    if t_norm < 1e-8:
        return None, None, mask, inliers, ratio, parallax

    t = t / t_norm

    return R, t, mask, inliers, ratio, parallax


# ============================================================
# TRIANGULATION
# ============================================================

def triangulate_points(p1, p2, R, t, K, D):

    pts1_n = undistort_points_fisheye(p1, K, D)
    pts2_n = undistort_points_fisheye(p2, K, D)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])

    pts4d = cv2.triangulatePoints(
        P1,
        P2,
        pts1_n.T,
        pts2_n.T
    )

    pts3d = pts4d[:3] / pts4d[3]

    depth1 = pts3d[2]

    pts3d_cam2 = R @ pts3d + t.reshape(3, 1)

    depth2 = pts3d_cam2[2]

    valid = (
        np.isfinite(pts3d).all(axis=0)
        & (depth1 > 0)
        & (depth2 > 0)
    )

    return pts3d[:, valid].T


# ============================================================
# VISUALIZATION
# ============================================================

def save_tracking_image(img, p1, p2, frame_id):

    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    a = p1.reshape(-1, 2)
    b = p2.reshape(-1, 2)

    for x, y in zip(a, b):

        x1, y1 = int(x[0]), int(x[1])
        x2, y2 = int(y[0]), int(y[1])

        cv2.circle(canvas, (x2, y2), 2, (0, 255, 0), -1)

        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 180, 255),
            1
        )

    cv2.imwrite(
        str(OUT / f"tracking_{frame_id:04d}.png"),
        canvas
    )


def plot_trajectory(vo_traj, gt):

    vo = np.asarray(vo_traj)

    plt.figure(figsize=(8, 8))

    plt.plot(
        vo[:, 0],
        vo[:, 2],
        linewidth=1.2,
        label="VO"
    )

    if len(gt) > 0:

        gt_xyz = np.array([g["t"] for g in gt])

        gt_xyz = gt_xyz - gt_xyz[0]

        plt.plot(
            gt_xyz[:, 0],
            gt_xyz[:, 2],
            linewidth=1.2,
            label="Ground Truth"
        )

    plt.xlabel("x")
    plt.ylabel("z")

    plt.title("VO Trajectory vs Ground Truth")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.savefig(str(OUT_PLOT), dpi=200)

    plt.close()


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(timestamps, trajectory, stats):

    traj = np.asarray(trajectory)

    with open(OUT_TRAJ, "w", encoding="utf-8") as f:

        for ts, p in zip(timestamps[:len(traj)], traj):

            f.write(
                f"{ts:.9f} "
                f"{p[0]:.6f} "
                f"{p[1]:.6f} "
                f"{p[2]:.6f}\n"
            )

    with open(OUT_STATS, "w", encoding="utf-8") as f:

        f.write(
            "frame,tracked,inliers,ratio,parallax,status\n"
        )

        for row in stats:

            f.write(
                "{},{},{},{:.6f},{:.6f},{}\n".format(*row)
            )


# ============================================================
# MAIN LOOP
# ============================================================

def run():

    K, D, T_cam_imu = load_calibration()

    paths, timestamps = load_images()

    gt = load_ground_truth()

    print("=== WEEK 3-4 IMPROVED VO FRONTEND ===")

    prev_img = read_img(paths[0])

    prev_pts = detect_orb_grid(
        prev_img,
        max_features=700
    )

    if prev_pts is None or len(prev_pts) < 150:
        raise RuntimeError("Initial feature detection failed")

    T_wc = np.eye(4, dtype=np.float64)

    trajectory = [T_wc[:3, 3].copy()]
    traj_ts = [timestamps[0]]

    stats = []

    successful = 0
    skipped = 0
    resets = 0

    for i in range(1, len(paths)):

        curr_img = read_img(paths[i])

        tracked_prev, tracked_curr = track_klt(
            prev_img,
            curr_img,
            prev_pts
        )

        if tracked_curr is None or len(tracked_curr) < 100:

            skipped += 1
            resets += 1

            new_pts = detect_orb_grid(
                curr_img,
                max_features=700
            )

            if new_pts is not None and len(new_pts) >= 150:

                prev_pts = new_pts
                prev_img = curr_img.copy()

            stats.append(
                (i, 0, 0, 0.0, 0.0, "RESET")
            )

            continue

        R, t, mask, inliers, ratio, parallax = estimate_motion(
            tracked_prev,
            tracked_curr,
            K,
            D
        )

        if R is None or t is None:

            skipped += 1

            prev_img = curr_img.copy()
            prev_pts = tracked_curr.copy()

            stats.append(
                (
                    i,
                    len(tracked_curr),
                    inliers,
                    ratio,
                    parallax,
                    "GEOM_SKIP"
                )
            )

            continue

        pts3d = triangulate_points(
            tracked_prev[mask],
            tracked_curr[mask],
            R,
            t,
            K,
            D
        )

        if len(pts3d) < 20:

            skipped += 1

            stats.append(
                (
                    i,
                    len(tracked_curr),
                    inliers,
                    ratio,
                    parallax,
                    "TRIANG_FAIL"
                )
            )

            continue

        T_rel = np.eye(4)

        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.reshape(3)

        T_wc = T_wc @ np.linalg.inv(T_rel)

        trajectory.append(
            T_wc[:3, 3].copy()
        )

        traj_ts.append(
            timestamps[i]
        )

        successful += 1

        stats.append(
            (
                i,
                len(tracked_curr),
                inliers,
                ratio,
                parallax,
                "OK"
            )
        )

        if i % 100 == 0:

            print(
                f"frame={i:04d} "
                f"inliers={inliers:03d} "
                f"ratio={ratio:.3f} "
                f"parallax={parallax:.4f} "
                f"triangulated={len(pts3d)}"
            )

            save_tracking_image(
                curr_img,
                tracked_prev,
                tracked_curr,
                i
            )

        prev_img = curr_img.copy()
        prev_pts = tracked_curr.copy()

        if len(prev_pts) < 180:

            new_pts = detect_orb_grid(
                curr_img,
                max_features=700
            )

            if new_pts is not None and len(new_pts) >= 150:
                prev_pts = new_pts

    save_outputs(traj_ts, trajectory, stats)

    plot_trajectory(trajectory, gt)

    print("\nDONE")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Resets: {resets}")


if __name__ == "__main__":
    run()