import csv
from pathlib import Path
import numpy as np
import cv2
import yaml

# =============================
# PATH CONFIG
# =============================
PROJECT_ROOT = Path(r"C:\Users\Admin\vio_project_shanza")

SEQ_NAME = "outdoors5_512_16"  # Change this to the actual sequence name you want to load
DATASET_NAME = "dataset-outdoors5_512_16"

DATASET = PROJECT_ROOT / "data" / SEQ_NAME / DATASET_NAME

MAV0 = DATASET / "mav0"
DSO = DATASET / "dso"

CAM0_DIR = MAV0 / "cam0"
IMU_DIR = MAV0 / "imu0"
GT_DIR = MAV0 / "mocap0"

IMG_DIR = CAM0_DIR / "data"
IMG_CSV = CAM0_DIR / "data.csv"
IMU_CSV = IMU_DIR / "data.csv"
GT_CSV = GT_DIR / "data.csv"
CALIB_FILE = DSO / "camchain.yaml"


# =============================
# LOADER CLASS
# =============================
class Loader:
    def __init__(self):
        self.K = None
        self.D = None
        self.dist_model = None
        self.T_cam_imu = None
        self.image_paths = []
        self.timestamps = []
        self.imu = []
        self.gt = []

    def load_calib(self):
        if not CALIB_FILE.exists():
            raise FileNotFoundError(f"Calibration file not found at {CALIB_FILE}")

        text = CALIB_FILE.read_text(encoding="utf-8")
        if text.startswith("%YAML:1.0"):
            text = text.replace("%YAML:1.0", "", 1).strip()

        data = yaml.safe_load(text)
        cam = data["cam0"]

        fx, fy, cx, cy = cam["intrinsics"]

        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_model = cam.get("distortion_model", "equidistant")
        self.D = np.array(
            cam.get("distortion_coeffs", [0, 0, 0, 0]),
            dtype=np.float64
        )

        if "T_cam_imu" in cam:
            self.T_cam_imu = np.array(cam["T_cam_imu"], dtype=np.float64)
        else:
            print("WARNING: T_cam_imu not found. VIO will not work correctly.")
            self.T_cam_imu = np.eye(4, dtype=np.float64)

    def load_images(self):
        if not IMG_CSV.exists():
            raise FileNotFoundError(f"Image CSV not found at {IMG_CSV}")

        with open(IMG_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if len(row) < 2:
                    continue

                ts = float(row[0]) * 1e-9
                path = IMG_DIR / row[1].strip()

                if path.exists():
                    self.timestamps.append(ts)
                    self.image_paths.append(str(path))

        if not self.image_paths:
            raise RuntimeError("No images found.")

    def load_imu(self):
        if not IMU_CSV.exists():
            raise FileNotFoundError(f"IMU CSV not found at {IMU_CSV}")

        with open(IMU_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if len(row) < 7:
                    continue

                ts = float(row[0]) * 1e-9
                gyro = np.array(row[1:4], dtype=np.float64)
                acc = np.array(row[4:7], dtype=np.float64)

                self.imu.append((ts, gyro, acc))

    def load_gt(self):
        if not GT_CSV.exists():
            print("Mocap GT file not found. This is normal for outdoors5 if only start/end GT is available.")
            return

        with open(GT_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if len(row) < 4:
                    continue

                ts = float(row[0]) * 1e-9
                pos = np.array(row[1:4], dtype=np.float64)
                self.gt.append((ts, pos))

    def load_all(self):
        self.load_calib()
        self.load_images()
        self.load_imu()
        self.load_gt()


# =============================
# SANITY CHECKS
# =============================
def check(loader):
    print("\n=== DATASET ===")
    print(f"Sequence: {SEQ_NAME}")
    print(f"Dataset path: {DATASET}")

    print("\n=== DATA STATS ===")
    print(f"Images: {len(loader.image_paths)}")
    print(f"IMU:    {len(loader.imu)}")
    print(f"GT:     {len(loader.gt)}")

    print("\n=== CALIBRATION ===")
    print(f"Model: {loader.dist_model}")
    print(f"K:\n{loader.K}")
    print(f"D: {loader.D}")
    print(f"T_cam_imu:\n{loader.T_cam_imu}")

    if len(loader.timestamps) > 0 and len(loader.imu) > 0:
        t_img = loader.timestamps[0]
        imu_ts = np.array([x[0] for x in loader.imu])
        idx = np.argmin(np.abs(imu_ts - t_img))
        sync_err = abs(imu_ts[idx] - t_img)
        print(f"\nIMU Sync Error: {sync_err:.6f} s")
    else:
        print("\nIMU Sync Error: unavailable")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    loader = Loader()
    loader.load_all()
    check(loader)