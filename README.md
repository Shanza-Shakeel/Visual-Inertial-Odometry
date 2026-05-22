# From Monocular Visual Odometry to Visual–Inertial Odometry on the TUM VI Dataset

This repository contains a classical geometry-based Visual Odometry (VO) and Visual–Inertial Odometry (VIO) pipeline developed and evaluated on the TUM VI dataset.

The project starts with a monocular visual odometry frontend based on sparse feature tracking, epipolar geometry, triangulation, PnP pose estimation, and motion-only bundle adjustment. It is then extended with IMU preintegration and an optimization-based visual–inertial backend.

The aim of this work is not to reproduce a complete state-of-the-art SLAM system. Instead, the goal is to implement the main VO/VIO concepts from the ground up and critically analyze their behavior, including tracking robustness, drift accumulation, scale ambiguity, and failure cases on indoor and outdoor sequences.

---

## 1. Project Motivation

Visual odometry estimates the motion of a camera from image sequences. It is important in robotics and computer vision when GPS or external positioning systems are unavailable, especially in indoor spaces, corridors, and mobile robot navigation.

A monocular camera is simple and lightweight, but it has a major limitation: it cannot directly recover metric scale. This means the estimated trajectory is only known up to an unknown scale factor. Over time, small errors in feature matching, pose estimation, and triangulation accumulate as drift.

Visual–inertial odometry attempts to reduce some of these weaknesses by combining visual information with IMU measurements. The IMU provides short-term acceleration and angular velocity information, which can improve local motion consistency. However, VIO is sensitive to initialization, gravity estimation, IMU bias handling, and the quality of the visual frontend.

This project investigates these issues using a compact classical implementation.

---

## 2. Main Contributions

This repository includes:

- A TUM VI dataset loader for images, IMU data, timestamps, calibration, and ground truth where available.
- A monocular VO frontend using ORB features, descriptor matching, KLT tracking, RANSAC, essential matrix estimation, and pose recovery.
- Sparse triangulation of 3D landmarks.
- PnP-based camera tracking with RANSAC.
- Motion-only bundle adjustment using nonlinear least-squares optimization.
- Sim(3)-based alignment for monocular VO evaluation.
- IMU preintegration between selected frames.
- A simplified optimization-based VIO backend.
- Evaluation using ATE, RPE, drift, tracking statistics, and qualitative trajectory analysis.
- Comparison with reference SLAM behavior such as ORB-SLAM3.
- Discussion of limitations, robustness issues, and failure cases.

---

## 3. Dataset

The experiments use the TUM VI dataset.

Main sequences:

| Sequence | Purpose |
|---|---|
| Room2 | Main quantitative evaluation using ground truth |
| Corridor3 | Robustness test in repeated indoor corridor structure |
| Outdoors5 | Long outdoor trajectory and drift analysis |

Expected dataset structure:

```text
data/
  room2/
    dataset-room2_512_16/
      mav0/
        cam0/
          data/
          data.csv
        imu0/
          data.csv
        mocap0/
          data.csv

  corridor3/
    dataset-corridor3_512_16/
      mav0/
        cam0/
        imu0/

  outdoors5/
    dataset-outdoors5_512_16/
      mav0/
        cam0/
        imu0/
## Techniques Used

This project uses a classical geometry-based VO/VIO pipeline. No deep learning, loop closure, or global SLAM map reuse is used.

### Visual Frontend

- **ORB feature detection**
  - Used to extract sparse image keypoints.
  - Chosen because it is fast and works with binary descriptors.

- **Grid-based feature distribution**
  - Used to avoid features being concentrated in only one image region.
  - Helps improve geometric stability.

- **Descriptor matching**
  - ORB descriptors are matched between frames.
  - Bad matches are reduced using ratio filtering and geometric checks.

- **KLT optical flow tracking**
  - Used for short-term feature tracking between consecutive frames.
  - Helps preserve feature continuity across nearby images.

- **RANSAC outlier rejection**
  - Used during essential matrix estimation and PnP.
  - Removes incorrect correspondences before pose estimation.

---

### Monocular VO Geometry

- **Fundamental matrix estimation**
  - Estimates the epipolar relation between two views.

- **Essential matrix computation**
  - Computed from the fundamental matrix using camera intrinsics:

```text
E = Kᵀ F K
