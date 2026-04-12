# Pocket2 RGBD Generation Pipeline

Offline processing pipeline for generating pinhole-posed-RGBD sequences from Pocket2 handheld scanning device.

## Directory Structure

pocket2_rgbd_gen_pipeline/
├── camera_pose_interpolation.py    # Camera pose interpolation (IMU pose → Camera pose)
├── undisort_test_pipeline.py       # Image undistortion (Fisheye → Pinhole)
├── pcd_image_projection_sequence.py # Point cloud depth projection (Sequential processing)
└── pcd_image_projection_parallel.py # Point cloud depth projection (Parallel processing)

## Processing Workflow

### 1. Camera Pose Interpolation

**Function**: Generate camera pose files from IMU poses and Til/Tcl extrinsics.

**Input**:
- `trigger_times.txt` - Image trigger timestamps
- `img_pos.txt` - IMU pose file
- `cam_in_ex_opt.txt` - Camera intrinsics and extrinsics (contains Til and Tcl matrices)

**Output**:
- `cam_0_pos.txt` - Camera 0 pose
- `cam_1_pos.txt` - Camera 1 pose
- `cam_2_pos.txt` - Camera 2 pose

**Format**: Each line `frame_id timestamp x y z qw qx qy qz`

**Usage**:
```bash
python camera_pose_interpolation.py
# Use GUI to select input files and execute
```

### 2. Image Undistortion

**Function**: Correct fisheye camera images to pinhole model images.

**Input**:
- `MT*/image/cam_X/` - Original fisheye image directory
- `cam_in_ex_opt.txt` - Camera intrinsics file (contains distortion coefficients k2-k7)

**Output**:
- `rectify_cam_X/` - Corrected image directory (640x480)
- `rectify_cam_X_param.txt` - Corrected intrinsics matrix K

**Usage**:
```bash
# Edit cam_name and path parameters in the script
cam_name = 'cam_1'  # Can be cam_0 or cam_1
img_dir = "/path/to/MTxxxx/image"

python undisort_test_pipeline.py
```

### 3. Point Cloud Depth Projection (Sequential)

**Function**: Project 3D point cloud onto 2D camera images, generating depth maps and visualization overlays.

**Input**:
- PCD file - Global point cloud (supports ASCII and binary formats)
- `cam_X_pos.txt` - Camera pose file
- Undistorted images - `rectify_cam_X/*.jpg`
- Camera intrinsics - `rectify_cam_X_param.txt`

**Output**:
- `depth/` - Depth map sequence (PNG format, unit: millimeters)
- `depth_vis/` - Depth visualization images
- `overlay/` - Point cloud and image overlay images

**Usage**:
```bash
python pcd_image_projection_sequence.py
# Use GUI to select PCD, pose, image directory and execute
```

### 4. Point Cloud Depth Projection (Parallel)

**Function**: Same as sequential version, but uses multiprocessing for parallel acceleration.

**Advantages**:
- Faster processing, suitable for large-scale data
- Supports shared memory for point cloud data transfer
- Adaptive downsampling for memory efficiency

**Usage**:
```bash
python pcd_image_projection_parallel.py
# Use GUI to select PCD, pose, image directory and execute
```

## Recommended Execution Order

1. **camera_pose_interpolation.py** → Generate camera poses
2. **undisort_test_pipeline.py** → Correct images to pinhole model
3. **pcd_image_projection_sequence.py** or **pcd_image_projection_parallel.py** → Generate depth maps

## Complete Data Directory Structure

```
MTxxxx/
├── MANIFOLD_MTxxxx-Cloud_Opt.pcd   # Global point cloud
├── image/
│   ├── cam_0/                       # Original images
│   ├── cam_1/
│   ├── cam_2/
│   ├── rectify_cam_0/               # Corrected images
│   ├── rectify_cam_0_param.txt      # Corrected intrinsics
│   ├── rectify_cam_1/
│   ├── rectify_cam_1_param.txt
│   ├── rectify_cam_2/
│   ├── rectify_cam_2_param.txt
│   └── cam_in_ex_opt.txt            # Camera intrinsics and extrinsics
├── cam_0_pos.txt                    # Camera poses
├── cam_1_pos.txt
├── cam_2_pos.txt
└── projection_results/              # Projection output
    ├── depth/
    ├── depth_vis/
    └── overlay/
```

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- psutil (optional, for memory adaptive downsampling)
- Open3D (optional, for voxel downsampling)
