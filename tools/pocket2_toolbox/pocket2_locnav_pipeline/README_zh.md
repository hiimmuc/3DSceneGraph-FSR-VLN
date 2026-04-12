# Pocket2 定位与导航管线

基于 Pocket2 手持扫描设备重建点云，适配定位与导航的离线处理流程。

## 目录结构

pocket2_loc_nav_pipeline/
├── pcd2keyframe.py        # 关键帧提取（Fast-LIVO2定位用）
├── pcd2pcd.py             # 全局地图降采样（Fast-LIVO2定位地图）
├── pcd2ply.py             # 点云格式转换（栅格地图生成前置输入）
├── grid_map_gen.py        # 栅格地图生成（Nav2导航用）
└── config_grid_map.json   # 栅格地图生成配置文件

## 外部数据文件

- `MTxxxx/image/img_pos_opt.txt` - 优化后的位姿文件（由 MindCloudApp 生成）
  - 格式：`frame_id timestamp x y z qw qx qy qz`
- `MTxxxx/map/keyframe_pose.txt` - 关键帧位姿（从 `MTxxxx/image/img_pos_opt.txt` 拷贝）

## 处理流程

### 1. 关键帧提取（Fast-LIVO2 定位）

**功能**：从全局点云中提取关键帧点云，每个关键帧包含该视角的点云数据。

**输入**：
- `MANIFOLD_MTxxxx-Cloud_Opt.pcd` - 全局点云（需包含 label 字段）
- `MTxxxx/map/keyframe_pose.txt` - 关键帧位姿文件（从 `MTxxxx/image/img_pos_opt.txt` 拷贝）

**输出**：
- `MTxxxx/map/keyframe_cloud/` - 关键帧点云目录（`xxxxxx.pcd` 格式）

**使用方法**：
```bash
python pcd2keyframe.py
# 或编辑脚本中的路径参数
```

### 2. 全局地图降采样

**功能**：对全局点云进行降采样，生成 Fast-LIVO2 定位用的匹配地图。

**输入**：
- `MANIFOLD_MTxxxx-Cloud_Opt.pcd` - 全局点云

**输出**：
- `MTxxxx/map/cloudGlobal.pcd` - 降采样后的全局地图

**使用方法**：
```bash
python pcd2pcd.py
# 编辑脚本中的路径和 voxel_size 参数
```

### 3. 点云格式转换（PLY）

**功能**：将 PCD 格式转换为 PLY 格式，用于栅格地图生成。

**输入**：
- `MANIFOLD_MTxxxx-Cloud_Opt.pcd` - 全局点云

**输出**：
- `MTxxxx/map/cloudGlobal.ply` - PLY 格式点云

**使用方法**：
```bash
python pcd2ply.py
# 编辑脚本中的路径和 voxel_size 参数
```

### 4. 栅格地图生成（Nav2 导航）

**功能**：将点云转换为 2D 栅格地图，用于 Nav2 导航。

**输入**：
- PLY 或 PCD 格式点云文件
- `config_grid_map.json` - 配置文件

**配置文件参数**：
- `origin_file_directory` - 点云文件目录
- `origin_file_name` - 点云文件名（不含扩展名）
- `rotation_matrix` - 旋转矩阵
- `thre_z_min / thre_z_max` - 高度滤波范围
- `flag_pass_through` - 是否启用直通滤波
- `thre_radius / thres_point_count` - 半径离群点滤波参数
- `map_resolution` - 栅格地图分辨率
- `map_file_name` - 输出地图文件名

**输出**：
- `.pgm` - 栅格地图图像
- `.yaml` - 地图元数据

**使用方法**：
```bash
python grid_map_gen.py
```

### [可选] 5. 栅格地图人工修图

生成的栅格地图可能存在一些误检或漏检区域，可使用 GIMP 进行人工修正。

**工具要求**：GIMP（开源图像编辑器）

**注意事项**：
- **不调整**图像分辨率和旋转角度
- **不修改** `grid_map.yaml` 文件
- 只修复 `.pgm` 文件中的障碍物栅格

**使用方法**：
1. 使用 GIMP 打开 `grid_map.pgm` 文件
2. 使用**铅笔工具**绘制障碍物（将对应区域涂黑）
3. 使用**橡皮擦工具**擦除误检区域（将对应区域涂白）
4. 保存修改后的 `.pgm` 文件

## 推荐执行顺序

1. 拷贝 `MTxxxx/image/img_pos_opt.txt` → `MTxxxx/map/keyframe_pose.txt`
2. **pcd2keyframe.py** → 准备定位关键帧
3. **pcd2pcd.py** → 准备定位地图
4. **pcd2ply.py** → 转换为 PLY 格式
5. **grid_map_gen.py** → 生成导航栅格地图
6. **[可选] 人工修图** → 使用 GIMP 修正栅格地图

## 完整数据目录结构

```
MTxxxx/
├── MANIFOLD_MTxxxx-Cloud_Opt.pcd   # 全局点云
├── image/
│   └── img_pos_opt.txt              # 优化后的位姿（MindCloudApp生成）
└── map/
    ├── keyframe_pose.txt            # 关键帧位姿（从img_pos_opt.txt拷贝）
    ├── cloudGlobal.pcd              # 降采样地图（Fast-LIVO2定位）
    ├── cloudGlobal.ply              # PLY格式地图（栅格地图输入）
    ├── keyframe_cloud/              # 关键帧点云
    │   ├── 000000.pcd
    │   ├── 000001.pcd
    │   └── ...
    └── grid_map.pgm / grid_map.yaml  # 栅格地图（Nav2导航）
```

## 定位导航地图部署

生成的栅格地图需要部署到机器人上才能进行导航。以下是部署配置步骤：

### 1. 地图文件放置

将生成的地图文件放置到导航配置中指定的路径：

```
/workspace/map/
├── grid_map/           # 导航栅格地图
│   ├── grid_map.yaml    # 地图元数据
│   └── grid_map.pgm     # 栅格地图图像
├── cloudGlobal.pcd      # 定位用降采样点云地图
├── keyframe_cloud/      # 关键帧点云目录
│   ├── 000000.pcd
│   ├── 000001.pcd
│   └── ...
└── keyframe_pose.txt    # 关键帧位姿文件
```

### 2. 配置文件修改

定位导航地图路径在 `navigation2.launch.py` 中配置：

**文件路径**：`nav_agent/humble_localization_nav2/g1_navigation2/launch/navigation2.launch.py`

**配置项**（第39行）：
```python
map_dir = LaunchConfiguration(
    'map',
    default=os.path.join('/workspace/map/grid_map/grid_map.yaml'))
```

### 3. 定位地图配置

Fast-LIVO2 定位功能需要在 `mid360_online_reloc.yaml` 中配置定位地图路径：

**文件路径**：`nav_agent/humble_localization_nav2/lio_mapping_loc/config/mid360_online_reloc.yaml`

**配置项**（第4行）：
```yaml
priorDir: "/workspace/map/"
```

### 4. 验证部署

启动导航节点，确认地图能够正常加载：
```bash
ros2 launch g1_navigation2 navigation2.launch.py
```

## 依赖

- Python 3.8+
- Open3D
- NumPy
- scikit-learn（用于最近邻搜索）
- PyYAML
- PIL（Pillow）
