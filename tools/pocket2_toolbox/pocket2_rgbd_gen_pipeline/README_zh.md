# Pocket2 RGBD 生成管线

基于 Pocket2 手持扫描设备生成 pinhole-posed-RGBD 序列的离线处理流程。

## 目录结构

pocket2_rgbd_gen_pipeline/
├── camera_pose_interpolation.py    # 相机位姿插值（IMU位姿→相机位姿）
├── undisort_test_pipeline.py       # 图像去畸变（鱼眼→针孔）
├── pcd_image_projection_sequence.py # 点云投影深度（串行处理）
└── pcd_image_projection_parallel.py # 点云投影深度（并行处理）

## 处理流程

### 1. 相机位姿插值

**功能**：根据 IMU 位姿和 Til/Tcl 外参，生成相机位姿文件。

**输入**：
- `trigger_times.txt` - 图像触发时间戳
- `img_pos.txt` - IMU 位姿文件
- `cam_in_ex_opt.txt` - 相机内外参（含 Til 和 Tcl 矩阵）

**输出**：
- `cam_0_pos.txt` - 相机0位姿
- `cam_1_pos.txt` - 相机1位姿
- `cam_2_pos.txt` - 相机2位姿

**格式**：每行 `frame_id timestamp x y z qw qx qy qz`

**使用方法**：
```bash
python camera_pose_interpolation.py
# 使用 GUI 界面选择输入文件并执行
```

### 2. 图像去畸变

**功能**：将鱼眼相机拍摄的图像校正为针孔模型图像。

**输入**：
- `MT*/image/cam_X/` - 原始鱼眼图像目录
- `cam_in_ex_opt.txt` - 相机内参文件（含畸变系数 k2-k7）

**输出**：
- `rectify_cam_X/` - 校正后图像目录（640x480）
- `rectify_cam_X_param.txt` - 校正后内参矩阵 K

**使用方法**：
```bash
# 编辑脚本中的 cam_name 和路径参数
cam_name = 'cam_1'  # 可改为 cam_0 或 cam_1
img_dir = "/path/to/MTxxxx/image"

python undisort_test_pipeline.py
```

### 3. 点云投影深度（串行版）

**功能**：将 3D 点云投影到 2D 相机图像，生成深度图和可视化叠加图。

**输入**：
- PCD 文件 - 全局点云（支持 ASCII 和二进制格式）
- `cam_X_pos.txt` - 相机位姿文件
- 去畸变后的图像 - `rectify_cam_X/*.jpg`
- 相机内参 - `rectify_cam_X_param.txt`

**输出**：
- `depth/` - 深度图序列（PNG格式，单位：毫米）
- `depth_vis/` - 深度可视化图
- `overlay/` - 点云与图像叠加图

**使用方法**：
```bash
python pcd_image_projection_sequence.py
# 使用 GUI 界面选择 PCD、位姿、图像目录并执行
```

### 4. 点云投影深度（并行版）

**功能**：与串行版相同，但使用多进程并行加速处理。

**优势**：
- 处理速度快，适合大规模数据
- 支持共享内存传输点云数据
- 自适应降采样保证内存效率

**使用方法**：
```bash
python pcd_image_projection_parallel.py
# 使用 GUI 界面选择 PCD、位姿、图像目录并执行
```

## 推荐执行顺序

1. **camera_pose_interpolation.py** → 生成相机位姿
2. **undisort_test_pipeline.py** → 校正图像为针孔模型
3. **pcd_image_projection_sequence.py** 或 **pcd_image_projection_parallel.py** → 生成深度图

## 完整数据目录结构

```
MTxxxx/
├── MANIFOLD_MTxxxx-Cloud_Opt.pcd   # 全局点云
├── image/
│   ├── cam_0/                       # 原始图像
│   ├── cam_1/
│   ├── cam_2/
│   ├── rectify_cam_0/               # 校正后图像
│   ├── rectify_cam_0_param.txt      # 校正后内参
│   ├── rectify_cam_1/
│   ├── rectify_cam_1_param.txt
│   ├── rectify_cam_2/
│   ├── rectify_cam_2_param.txt
│   └── cam_in_ex_opt.txt            # 相机内外参
├── cam_0_pos.txt                    # 相机位姿
├── cam_1_pos.txt
├── cam_2_pos.txt
└── projection_results/              # 投影输出
    ├── depth/
    ├── depth_vis/
    └── overlay/
```

## 依赖

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- psutil（可选，用于内存自适应降采样）
- Open3D（可选，用于体素降采样）
