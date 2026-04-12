def statistical_outlier_removal(pcd, nb_neighbors=30, std_ratio=1.0):
    print(f'[grid_map_gen] 统计离群点滤波: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}')
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    print(f'[grid_map_gen] 统计滤波后点数: {len(pcd.points)}')
    return pcd
import os
import json
import numpy as np
import open3d as o3d
import yaml
from PIL import Image

def load_config(config_path):
    print('[grid_map_gen] 加载配置文件:', config_path)
    with open(config_path, 'r') as f:
        return json.load(f)

def load_point_cloud(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f'[grid_map_gen] 加载点云文件: {file_path}')
    if ext == '.ply':
        return o3d.io.read_point_cloud(file_path)
    elif ext == '.pcd':
        return o3d.io.read_point_cloud(file_path)
    else:
        raise ValueError(f'Unsupported point cloud format: {ext}')

def apply_rotation(pcd, rotation_matrix):
    print('[grid_map_gen] 应用旋转矩阵')
    R = np.array(rotation_matrix)
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

def pass_through_filter(pcd, z_min, z_max, negative=False):
    print(f'[grid_map_gen] 直通滤波: z_min={z_min}, z_max={z_max}, negative={negative}')
    points = np.asarray(pcd.points)
    if not negative:
        mask = (points[:,2] >= z_min) & (points[:,2] <= z_max)
    else:
        mask = (points[:,2] < z_min) | (points[:,2] > z_max)
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    print(f'[grid_map_gen] 直通滤波后点数: {len(pcd.points)}')
    return pcd

def radius_outlier_filter(pcd, radius, min_points):
    print(f'[grid_map_gen] 半径离群点滤波: radius={radius}, min_points={min_points}')
    _, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    pcd = pcd.select_by_index(ind)
    print(f'[grid_map_gen] 半径滤波后点数: {len(pcd.points)}')
    return pcd

def pointcloud_to_gridmap(pcd, resolution):
    print(f'[grid_map_gen] 点云转栅格地图, 分辨率: {resolution}')
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        raise ValueError('No points left after filtering!')
    x_min, y_min = points[:,0].min(), points[:,1].min()
    x_max, y_max = points[:,0].max(), points[:,1].max()
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))
    print(f'[grid_map_gen] 地图尺寸: width={width}, height={height}')
    # -1: unknown, 0~100: occupancy value
    grid = np.full((height, width), -1, dtype=np.int16)
    for pt in points:
        i = int(np.floor((pt[0] - x_min) / resolution))
        j = int(np.floor((pt[1] - y_min) / resolution))
        if 0 <= i < width and 0 <= j < height:
            grid[j, i] = 100  # occupied
    return grid, (x_min, y_min), width, height

def save_pgm(grid, filename):
    print(f'[grid_map_gen] 保存PGM地图: {filename}')
    # 读取阈值
    # 这里假设 grid 传入时已是 occupancy 值（-1, 0~100），需在 main 里传入 config
    # 但 PIL 只支持8位灰度，所以先做三值映射
    # 这里直接用 main 里的 config 变量
    global config
    free_thresh = int(round(config['map_free_thresh'] * 100))
    occupied_thresh = int(round(config['map_occupied_thresh'] * 100))
    height, width = grid.shape
    pgm = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            v = grid[height - 1 - y, x]  # y轴翻转
            if v < 0 or v > 100:
                pgm[y, x] = 255  # unknown
            elif v <= free_thresh:
                pgm[y, x] = 255  # free
            elif v >= occupied_thresh:
                pgm[y, x] = 0    # occupied
            else:
                pgm[y, x] = 255  # unknown
    img = Image.fromarray(pgm, mode='L')
    img.save(filename)

def save_yaml(yaml_path, image_path, resolution, origin, occupied_thresh, free_thresh):
    print(f'[grid_map_gen] 保存YAML元数据: {yaml_path}')
    data = {
        'image': os.path.basename(image_path),
        'resolution': float(resolution),
        'origin': [float(origin[0]), float(origin[1]), 0.0],
        'negate': 0,
        'occupied_thresh': float(occupied_thresh),
        'free_thresh': float(free_thresh),
        'mode': 'trinary'
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def main():
    global config
    print('[grid_map_gen] 开始生成栅格地图...')
    config = load_config('/mnt/disk1/mapvln/pocket2/pocket2_pipeline/config_grid_map.json')
    file_dir = config['origin_file_directory']
    file_name = config['origin_file_name']
    pcd_path = os.path.join(file_dir, file_name + '.ply')
    if not os.path.exists(pcd_path):
        pcd_path = os.path.join(file_dir, file_name + '.pcd')
    if not os.path.exists(pcd_path):
        raise FileNotFoundError('Neither .ply nor .pcd file found!')
    pcd = load_point_cloud(pcd_path)
    pcd = apply_rotation(pcd, config['rotation_matrix'])
    # 统计离群点滤波，参数与C++一致
    pcd = statistical_outlier_removal(pcd, nb_neighbors=30, std_ratio=1.0)
    pcd = pass_through_filter(pcd, config['thre_z_min'], config['thre_z_max'], bool(config['flag_pass_through']))
    pcd = radius_outlier_filter(pcd, config['thre_radius'], config['thres_point_count'])
    grid, origin, width, height = pointcloud_to_gridmap(pcd, config['map_resolution'])
    map_file_base = config['map_file_name']
    pgm_path = map_file_base + '.pgm'
    yaml_path = map_file_base + '.yaml'
    save_pgm(grid, pgm_path)
    save_yaml(yaml_path, pgm_path, config['map_resolution'], origin, config['map_occupied_thresh'], config['map_free_thresh'])
    print(f'[grid_map_gen] 栅格地图已保存: {pgm_path}, {yaml_path}')

if __name__ == '__main__':
    main()
