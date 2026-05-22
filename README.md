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

```## Repository Structure

The repository is organized by sequence. Each sequence folder contains the Python scripts used at different project stages, while the corresponding results folder stores the generated outputs.

```text
.
├── Room2/
│   ├── week1.py
│   ├── week3.py
│   ├── week5.py
│   ├── week7.py
│   ├── week8.py
│   └── week11.py
│
├── room2 Results/
│   ├── week3_4_improved/
│   ├── week5_6_strong_reset_debug/
│   ├── week7_vo_eval/
│   ├── week8_10_posegraph_vio_rotation_refined/
│   └── week11_final_evaluation/
│
├── corridoor3/corridor3/
│   ├── week1.py
│   ├── week3.py
│   ├── week5.py
│   ├── week7.py
│   ├── week8.py
│   └── week11.py
│
├── corridor3 Results/
│   ├── week3_4_improved/
│   ├── week5_6_strong_reset_debug/
│   ├── week7_vo_eval/
│   ├── week8_10_posegraph_vio_rotation_refined/
│   └── week11_final_evaluation/
│
├── outdoor5/outdoor5/
│   ├── week1.py
│   ├── week3.py
│   ├── week5.py
│   ├── week7.py
│   ├── week8.py
│   └── week11.py
│
├── outdoor5 Results/
│   ├── week3_4_improved/
│   ├── week5_6_strong_reset_debug/
│   ├── week7_vo_eval/
│   ├── week8_10_posegraph_vio_rotation_refined/
│   └── week11_final_evaluation/
│
└── README.md
```





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
```
- **Pose recovery from the essential matrix**
  - Recovers relative camera rotation and translation direction between frames.
  - Translation is recovered only up to scale.

- **Chirality / positive depth check**
  - Used to select the physically valid pose configuration after pose recovery.

- **Sparse triangulation**
  - Reconstructs sparse 3D landmarks from matched feature correspondences.

- **PnP pose estimation with RANSAC**
  - Estimates camera pose from 2D–3D correspondences.
  - Used after initialization for long-term camera tracking.

- **Motion-only bundle adjustment**
  - Refines the estimated camera pose by minimizing reprojection error.
  - Improves local trajectory consistency.

---

### Scale Handling and Evaluation

- **Monocular scale ambiguity**
  - Monocular VO cannot directly recover metric scale.
  - Estimated trajectories are therefore only correct up to scale.

- **Sim(3) trajectory alignment**
  - Used to align estimated trajectories with ground truth for evaluation.

```text
p_aligned = s R p_est + t
```

where:

- `s` = scale factor  
- `R` = rotation alignment  
- `t` = translation offset  

- **Absolute Trajectory Error (ATE)**
  - Measures global trajectory accuracy after alignment.

- **Relative Pose Error (RPE)**
  - Measures short-term relative motion consistency.

- **Start–end drift analysis**
  - Used especially for long sequences where accumulated drift becomes significant.

---

### Visual–Inertial Backend

- **IMU preintegration**
  - Integrates inertial measurements between selected keyframes.
  - Produces relative inertial motion constraints:

```text
ΔR_ij
Δv_ij
Δp_ij
```

- **Sliding-window visual–inertial optimization**
  - Combines visual constraints and IMU constraints inside nonlinear optimization.
  - Refines pose, velocity, scale, gravity direction, and bias-related terms.

- **Gravity and scale initialization**
  - Multiple gravity candidates and scale checks are tested to improve optimization stability.

- **Bias and noise handling**
  - Simplified IMU bias and noise modeling is used during optimization.

---

### Robustness and Failure Handling

- **Frame skipping**
  - Used when tracking quality becomes unreliable.

- **Fallback essential matrix estimation**
  - Used when PnP tracking becomes unstable.

- **Landmark rebuilding**
  - Reconstructs landmarks when active tracking quality drops.

- **Strong reset logic**
  - Reinitializes tracking after severe pose estimation failure.

- **Tracking diagnostics**
  - Logs PnP successes, fallback counts, resets, failed frames, runtime, and FPS statistics.

---

### What This Pipeline Does Not Include

This implementation intentionally does not include:

- Deep learning methods
- CNN-based feature extraction
- Loop closure
- Relocalization
- Global map optimization
- Full SLAM map reuse
- Production-level IMU calibration

The project focuses on understanding the core geometry, tracking, optimization, and IMU preintegration components of classical VO/VIO systems.

---

## 5. Evaluation

The evaluation focuses on both quantitative accuracy and qualitative trajectory behavior.

### Quantitative Evaluation

Room2 is evaluated using:

- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- Sim(3) alignment
- Drift percentage

### Qualitative Evaluation

Corridor3 and Outdoors5 are analyzed using:

- Tracking robustness
- Failure frequency
- Reset behavior
- Drift accumulation
- Trajectory consistency
- VO vs VIO comparison
- Comparison with ORB-SLAM3

---

## 6. Runtime Statistics

| Sequence | Runtime | Processed Frames | FPS |
|---|---:|---:|---:|
| Room2 | 198.20 s | 2864 | 14.45 |
| Corridor3 | 401.34 s | 5775 | 14.39 |
| Outdoors5 | 968.32 s | 17740 | 18.32 |

---

## 7. Main Observations

- Room2 produced the most stable quantitative evaluation.
- Corridor3 was difficult because repeated corridor structure weakened feature correspondence stability.
- Outdoors5 remained locally trackable but accumulated strong global drift over long motion.
- IMU integration improved local motion consistency but did not fully correct long-term drift.
- VIO performance depended strongly on VO initialization quality, gravity estimation, and bias handling.
- The absence of loop closure and global optimization caused accumulated drift over time.

---

## 8. Repository Structure

```text
.
├── src/
│   ├── dataset_loader.py
│   ├── vo_frontend.py
│   ├── imu_preintegration.py
│   ├── vio_backend.py
│   ├── evaluation.py
│   └── utils/
│
├── data/
│
├── results/
│
├── figures/
│
├── report/
│
├── requirements.txt
└── README.md
```

---

## 9. Dependencies

Main libraries used:

```text
numpy
opencv-python
scipy
matplotlib
pyyaml
open3d
```

Install dependencies:

```bash
pip install numpy opencv-python scipy matplotlib pyyaml open3d
```

---

## 10. Reproducibility

Experiments were executed using:

- Python 3.x
- OpenCV
- NumPy
- SciPy
- Windows-based system
- Intel CPU
- No GPU acceleration

For reproducibility:

```python
np.random.seed(0)
```

The provided TUM VI calibration files were used directly without recalibration.

---

## 11. Authors

- Shanza Shakeel
- Alina Garcia
