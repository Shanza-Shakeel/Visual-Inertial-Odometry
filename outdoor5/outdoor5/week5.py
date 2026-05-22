import csv
from pathlib import Path
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt

try:
    from scipy.optimize import least_squares
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ================= PATHS =================
ROOT = Path(r"C:\Users\Admin\vio_project_shanza")
SEQ_NAME = "outdoors5_512_16"  # Change this to the actual sequence name you want to load
SEQ = ROOT / "data" / SEQ_NAME / "dataset-outdoors5_512_16"

IMG_DIR = SEQ / "mav0" / "cam0" / "data"
IMG_CSV = SEQ / "mav0" / "cam0" / "data.csv"
CALIB = SEQ / "dso" / "camchain.yaml"

OUT = ROOT / "result" / "week5_6_strong_reset_debug_outdoors5"
OUT.mkdir(parents=True, exist_ok=True)

OUT_TRAJ = OUT / "trajectory_tum.txt"
OUT_STATS = OUT / "stats_debug.csv"
OUT_PLOT = OUT / "trajectory_xz.png"


# ================= PARAMS =================
# Outdoors5 is longer and harder than room2.
MIN_TRACKS = 120
MIN_PNP = 35

REBUILD_IF_LESS = 180
REBUILD_EVERY = 20

E_THRESH = 0.006
PNP_THRESH = 0.025
MAX_TRI_ERR = 0.050

MIN_DEPTH = 0.05
MAX_DEPTH = 250.0

RESET_IF_LANDMARKS_BELOW = 100
RESET_AFTER_CONSEC_FAILS = 5
MAX_RESET_GAP = 35

GRID_FEATURES = 1400


# ================= LOAD =================
def read_yaml(path):
    text = path.read_text(encoding="utf-8")
    if text.startswith("%YAML:1.0"):
        text = text.replace("%YAML:1.0", "", 1).lstrip()
    return yaml.safe_load(text)


def load_calib():
    cam = read_yaml(CALIB)["cam0"]
    fx, fy, cx, cy = cam["intrinsics"]

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    D = np.array(
        cam.get("distortion_coeffs", [0, 0, 0, 0]),
        dtype=np.float64
    ).reshape(-1, 1)

    return K, D


def load_images():
    paths, ts = [], []

    with open(IMG_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) < 2:
                continue

            p = IMG_DIR / row[1].strip()
            if p.exists():
                paths.append(str(p))
                ts.append(float(row[0]) * 1e-9)

    if len(paths) < 50:
        raise RuntimeError("Too few images loaded.")

    return paths, ts


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    return cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(img)


# ================= FEATURES =================
def detect_grid(img, max_features=GRID_FEATURES):
    h, w = img.shape

    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=7
    )

    pts = []
    rows, cols = 4, 4
    per_cell = max_features // (rows * cols)

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols

            kps = orb.detect(img[y0:y1, x0:x1], None)
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[:per_cell]

            for kp in kps:
                x = kp.pt[0] + x0
                y = kp.pt[1] + y0

                if 5 <= x < w - 5 and 5 <= y < h - 5:
                    pts.append([x, y])

    if len(pts) == 0:
        return None

    return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)


def track_klt(img1, img2, pts1):
    if pts1 is None or len(pts1) == 0:
        return None, None, None

    pts2, st1, _ = cv2.calcOpticalFlowPyrLK(
        img1,
        img2,
        pts1,
        None,
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01)
    )

    if pts2 is None or st1 is None:
        return None, None, None

    back, st2, _ = cv2.calcOpticalFlowPyrLK(
        img2,
        img1,
        pts2,
        None,
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01)
    )

    if back is None or st2 is None:
        return None, None, None

    p1 = pts1.reshape(-1, 2)
    p2 = pts2.reshape(-1, 2)
    pb = back.reshape(-1, 2)

    fb = np.linalg.norm(p1 - pb, axis=1)
    h, w = img2.shape

    good = (
        (st1.ravel() == 1)
        & (st2.ravel() == 1)
        & (fb < 2.0)
        & np.isfinite(p2).all(axis=1)
        & (p2[:, 0] >= 5)
        & (p2[:, 0] < w - 5)
        & (p2[:, 1] >= 5)
        & (p2[:, 1] < h - 5)
    )

    return (
        pts1[good].reshape(-1, 1, 2),
        pts2[good].reshape(-1, 1, 2),
        good
    )


# ================= GEOMETRY =================
def undist_norm(pts, K, D):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)

    out = cv2.fisheye.undistortPoints(
        pts,
        K,
        D,
        R=np.eye(3),
        P=np.eye(3)
    )

    return out.reshape(-1, 2)


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def essential_pose(p1, p2, K, D):
    if p1 is None or p2 is None or len(p1) < MIN_TRACKS:
        return None, None, None

    n1 = undist_norm(p1, K, D)
    n2 = undist_norm(p2, K, D)

    E, mask = cv2.findEssentialMat(
        n1,
        n2,
        focal=1.0,
        pp=(0, 0),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=E_THRESH
    )

    if E is None or mask is None:
        return None, None, None

    mask = mask.ravel().astype(bool)

    if mask.sum() < 80:
        return None, None, None

    _, R, t, _ = cv2.recoverPose(
        E,
        n1[mask],
        n2[mask],
        focal=1.0,
        pp=(0, 0)
    )

    t = t.reshape(3)
    t = t / (np.linalg.norm(t) + 1e-12)

    return R, t, mask


def triangulate(T1_cw, T2_cw, p1, p2, K, D):
    n1 = undist_norm(p1, K, D)
    n2 = undist_norm(p2, K, D)

    Xh = cv2.triangulatePoints(
        T1_cw[:3],
        T2_cw[:3],
        n1.T,
        n2.T
    ).T

    X = Xh[:, :3] / (Xh[:, 3:4] + 1e-12)

    Xc1 = (T1_cw[:3, :3] @ X.T + T1_cw[:3, 3:4]).T
    Xc2 = (T2_cw[:3, :3] @ X.T + T2_cw[:3, 3:4]).T

    z1 = Xc1[:, 2]
    z2 = Xc2[:, 2]

    proj1 = Xc1[:, :2] / (Xc1[:, 2:3] + 1e-12)
    proj2 = Xc2[:, :2] / (Xc2[:, 2:3] + 1e-12)

    err = 0.5 * (
        np.linalg.norm(proj1 - n1, axis=1)
        + np.linalg.norm(proj2 - n2, axis=1)
    )

    good = (
        np.isfinite(X).all(axis=1)
        & (z1 > MIN_DEPTH)
        & (z2 > MIN_DEPTH)
        & (z1 < MAX_DEPTH)
        & (z2 < MAX_DEPTH)
        & (err < MAX_TRI_ERR)
    )

    return X[good].astype(np.float64), p2[good].astype(np.float32)


def build_map(img1, img2, T1_cw, K, D):
    pts1 = detect_grid(img1)

    if pts1 is None or len(pts1) < MIN_TRACKS:
        return None, None, None

    p1, p2, _ = track_klt(img1, img2, pts1)

    R, t, mask = essential_pose(p1, p2, K, D)

    if R is None:
        return None, None, None

    p1 = p1[mask].reshape(-1, 1, 2)
    p2 = p2[mask].reshape(-1, 1, 2)

    T21 = make_T(R, t)
    T2_cw = T21 @ T1_cw

    Xw, pts2 = triangulate(T1_cw, T2_cw, p1, p2, K, D)

    if len(Xw) < MIN_PNP:
        return None, None, None

    return T2_cw, Xw, pts2


def strong_reset(paths, start_i, prev_T, K, D):
    img_start = read_img(paths[start_i])

    for gap in range(3, MAX_RESET_GAP + 1):
        j = start_i + gap

        if j >= len(paths):
            break

        img_j = read_img(paths[j])

        Tj, Xnew, ptsnew = build_map(img_start, img_j, prev_T, K, D)

        if Tj is not None and Xnew is not None and ptsnew is not None and len(Xnew) >= MIN_PNP:
            return j, img_j, Tj, Xnew, ptsnew

    return None, None, None, None, None


# ================= PNP + BA + DIAGNOSTICS =================
def reproj_diagnostics(Xw, pts_px, R, t, K, D):
    Xc = (R @ Xw.T + t.reshape(3, 1)).T
    valid = Xc[:, 2] > 1e-8

    if valid.sum() == 0:
        return 999.0, 999.0, 999.0, 0

    Xc_valid = Xc[valid]
    pts_px_valid = pts_px.reshape(-1, 2)[valid]

    proj_norm = Xc_valid[:, :2] / Xc_valid[:, 2:3]
    obs_norm = undist_norm(pts_px_valid.reshape(-1, 1, 2), K, D)

    err_norm = np.linalg.norm(proj_norm - obs_norm, axis=1)

    proj_px = cv2.fisheye.distortPoints(
        proj_norm.reshape(-1, 1, 2).astype(np.float64),
        K,
        D
    ).reshape(-1, 2)

    err_px = np.linalg.norm(proj_px - pts_px_valid, axis=1)

    return (
        float(np.median(err_norm)),
        float(np.median(err_px)),
        float(np.max(err_px)),
        int(valid.sum())
    )


def ba_pose(Xw, obs_norm, rvec, tvec):
    def residual(x):
        R, _ = cv2.Rodrigues(x[:3])
        t = x[3:6]

        Xc = (R @ Xw.T + t.reshape(3, 1)).T
        valid = Xc[:, 2] > 1e-8

        proj = np.zeros((len(Xw), 2), dtype=np.float64)
        proj[valid] = Xc[valid, :2] / Xc[valid, 2:3]

        res = (proj - obs_norm).reshape(-1)
        res[~np.repeat(valid, 2)] = 10.0

        return res

    x0 = np.hstack([rvec.reshape(3), tvec.reshape(3)])

    res = least_squares(
        residual,
        x0,
        loss="huber",
        f_scale=0.015,
        max_nfev=20
    )

    return res.x[:3].reshape(3, 1), res.x[3:6].reshape(3, 1)


def solve_pnp(Xw, pts, K, D):
    if Xw is None or pts is None or len(Xw) < MIN_PNP:
        return None, None, 0, 999.0, 999.0, 999.0, 0

    obs_norm = undist_norm(pts, K, D).astype(np.float64)
    K_id = np.eye(3, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        Xw.astype(np.float64),
        obs_norm.reshape(-1, 1, 2),
        K_id,
        None,
        iterationsCount=120,
        reprojectionError=PNP_THRESH,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok or inliers is None or len(inliers) < MIN_PNP:
        return None, None, 0, 999.0, 999.0, 999.0, 0

    inliers = inliers.ravel()
    X_in = Xw[inliers]
    obs_in_norm = obs_norm[inliers]
    pts_in_px = pts.reshape(-1, 2)[inliers]

    ok, rvec, tvec = cv2.solvePnP(
        X_in.astype(np.float64),
        obs_in_norm.reshape(-1, 1, 2),
        K_id,
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        return None, None, 0, 999.0, 999.0, 999.0, 0

    if SCIPY_OK and len(X_in) >= 40:
        rvec, tvec = ba_pose(X_in, obs_in_norm, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    T_cw = make_T(R, t)

    err_norm, err_px_med, err_px_max, valid_depth = reproj_diagnostics(
        X_in,
        pts_in_px,
        R,
        t,
        K,
        D
    )

    return T_cw, inliers, len(inliers), err_norm, err_px_med, err_px_max, valid_depth


# ================= SAVE =================
def rot_to_quat(R):
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


def save(ts_used, poses_cw, stats):
    with open(OUT_TRAJ, "w", encoding="utf-8") as f:
        for ts, T_cw in zip(ts_used, poses_cw):
            T_wc = np.linalg.inv(T_cw)
            p = T_wc[:3, 3]
            q = rot_to_quat(T_wc[:3, :3])

            f.write(
                f"{ts:.9f} "
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write(
            "frame,tracked,pnp_inliers,"
            "err_norm,err_px_med,err_px_max,valid_depth,"
            "landmarks,status\n"
        )

        for row in stats:
            f.write(
                "{},{},{},{:.6f},{:.6f},{:.6f},{},{},{}\n".format(*row)
            )

    xyz = np.array([np.linalg.inv(T)[:3, 3] for T in poses_cw])

    plt.figure(figsize=(8, 8))
    plt.plot(xyz[:, 0], xyz[:, 2])
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Week 5-6 Iterative PnP + BA VO + Strong Reset - outdoors5")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(str(OUT_PLOT), dpi=200)
    plt.close()


def fail_row(i, tracked, landmarks, status):
    return (i, tracked, 0, 999.0, 999.0, 999.0, 0, landmarks, status)


# ================= MAIN =================
def run():
    K, D = load_calib()
    paths, timestamps = load_images()

    print("=== WEEK 5-6 ITERATIVE PNP + BA VO + STRONG RESET DEBUG - OUTDOORS5 ===")
    print("Frames:", len(paths))
    print("SciPy BA:", "ON" if SCIPY_OK else "OFF")

    img0 = read_img(paths[0])
    T0 = np.eye(4, dtype=np.float64)

    print("Searching init gap...")
    init = None

    for gap in range(3, 51):
        img_gap = read_img(paths[gap])
        Tgap, Xw, pts = build_map(img0, img_gap, T0, K, D)

        if Tgap is not None:
            print(f"INIT OK: 0 -> {gap}, landmarks={len(Xw)}")
            init = gap, img_gap, Tgap, Xw, pts
            break

        print(f"INIT failed: 0 -> {gap}")

    if init is None:
        raise RuntimeError("Initialization failed.")

    init_idx, prev_img, prev_T, Xw, prev_pts = init

    poses = [T0.copy(), prev_T.copy()]
    ts_used = [timestamps[0], timestamps[init_idx]]

    stats = [
        (0, 0, 0, 0.0, 0.0, 0.0, 0, 0, "INIT0"),
        (init_idx, len(prev_pts), 0, 0.0, 0.0, 0.0, 0, len(Xw), "INIT_OK")
    ]

    last_img = img0
    last_T = T0.copy()

    pnp_ok = 0
    fallback_ok = 0
    strong_reset_ok = 0
    total_fail = 0
    consecutive_fail = 0

    i = init_idx + 1

    while i < len(paths):
        curr_img = read_img(paths[i])
        _, curr_pts, mask = track_klt(prev_img, curr_img, prev_pts)

        if curr_pts is None or mask is None:
            Tnew, Xnew, ptsnew = build_map(prev_img, curr_img, prev_T, K, D)

            if Tnew is not None:
                poses.append(Tnew.copy())
                ts_used.append(timestamps[i])

                prev_img = curr_img
                prev_T = Tnew
                Xw = Xnew
                prev_pts = ptsnew

                fallback_ok += 1
                consecutive_fail = 0
                stats.append(fail_row(i, 0, len(Xw), "E_FALLBACK_TRACK"))

                i += 1
                continue

            total_fail += 1
            consecutive_fail += 1
            stats.append(fail_row(i, 0, len(Xw), "TRACK_FAIL"))

            if consecutive_fail >= RESET_AFTER_CONSEC_FAILS or len(Xw) < RESET_IF_LANDMARKS_BELOW:
                j, img_j, Tj, Xreset, ptsreset = strong_reset(paths, i, prev_T, K, D)

                if Tj is not None:
                    poses.append(Tj.copy())
                    ts_used.append(timestamps[j])

                    last_img = curr_img
                    last_T = prev_T.copy()

                    prev_img = img_j
                    prev_T = Tj.copy()
                    Xw = Xreset
                    prev_pts = ptsreset

                    fallback_ok += 1
                    strong_reset_ok += 1
                    consecutive_fail = 0

                    stats.append(fail_row(j, len(ptsreset), len(Xw), "STRONG_RESET_TRACK"))
                    print(f"STRONG RESET OK: {i} -> {j}, landmarks={len(Xw)}")

                    i = j + 1
                    continue

            i += 1
            continue

        Xtr = Xw[mask]

        (
            Tnew,
            inliers,
            nin,
            err_norm,
            err_px_med,
            err_px_max,
            valid_depth
        ) = solve_pnp(Xtr, curr_pts, K, D)

        if Tnew is not None:
            Xw = Xtr[inliers]
            prev_pts = curr_pts[inliers].reshape(-1, 1, 2)

            status = "ITER_PNP_OK"
            pnp_ok += 1
            consecutive_fail = 0

            if len(Xw) < REBUILD_IF_LESS or i % REBUILD_EVERY == 0:
                Ttmp, Xnew, ptsnew = build_map(last_img, curr_img, last_T, K, D)

                if Ttmp is not None and len(Xnew) > MIN_PNP:
                    Xw = Xnew
                    prev_pts = ptsnew
                    status = "ITER_PNP_OK_REBUILD"

            poses.append(Tnew.copy())
            ts_used.append(timestamps[i])

            stats.append((
                i,
                len(curr_pts),
                nin,
                err_norm,
                err_px_med,
                err_px_max,
                valid_depth,
                len(Xw),
                status
            ))

            if i % 100 == 0:
                print(
                    f"frame={i:05d} "
                    f"inliers={nin} "
                    f"err_norm={err_norm:.6f} "
                    f"err_px_med={err_px_med:.3f} "
                    f"err_px_max={err_px_max:.3f} "
                    f"landmarks={len(Xw)} "
                    f"{status}"
                )

            last_img = prev_img
            last_T = prev_T.copy()

            prev_img = curr_img
            prev_T = Tnew.copy()

            i += 1

        else:
            Tfb, Xnew, ptsnew = build_map(prev_img, curr_img, prev_T, K, D)

            if Tfb is not None:
                poses.append(Tfb.copy())
                ts_used.append(timestamps[i])

                Xw = Xnew
                prev_pts = ptsnew

                last_img = prev_img
                last_T = prev_T.copy()

                prev_img = curr_img
                prev_T = Tfb.copy()

                fallback_ok += 1
                consecutive_fail = 0
                stats.append(fail_row(i, len(curr_pts), len(Xw), "E_FALLBACK_PNP"))

                i += 1
                continue

            total_fail += 1
            consecutive_fail += 1
            stats.append(fail_row(i, len(curr_pts), len(Xw), "PNP_FAIL_TOTAL"))

            if consecutive_fail >= RESET_AFTER_CONSEC_FAILS or len(Xw) < RESET_IF_LANDMARKS_BELOW:
                j, img_j, Tj, Xreset, ptsreset = strong_reset(paths, i, prev_T, K, D)

                if Tj is not None:
                    poses.append(Tj.copy())
                    ts_used.append(timestamps[j])

                    last_img = curr_img
                    last_T = prev_T.copy()

                    prev_img = img_j
                    prev_T = Tj.copy()
                    Xw = Xreset
                    prev_pts = ptsreset

                    fallback_ok += 1
                    strong_reset_ok += 1
                    consecutive_fail = 0

                    stats.append(fail_row(j, len(ptsreset), len(Xw), "STRONG_RESET_PNP"))
                    print(f"STRONG RESET OK: {i} -> {j}, landmarks={len(Xw)}")

                    i = j + 1
                    continue

            i += 1

        if i % 500 == 0:
            print(
                f"summary frame={i:05d} "
                f"pnp_ok={pnp_ok} "
                f"fallback={fallback_ok} "
                f"strong_reset={strong_reset_ok} "
                f"fail={total_fail} "
                f"landmarks={len(Xw)}"
            )

    save(ts_used, poses, stats)

    print("\nDONE")
    print("Input frames:", len(paths))
    print("Saved poses:", len(poses))
    print("Iterative PnP OK:", pnp_ok)
    print("Essential fallback OK:", fallback_ok)
    print("Strong reset OK:", strong_reset_ok)
    print("Total failed:", total_fail)
    print("Trajectory:", OUT_TRAJ)
    print("Stats:", OUT_STATS)
    print("Plot:", OUT_PLOT)


if __name__ == "__main__":
    run()