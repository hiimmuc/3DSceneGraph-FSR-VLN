import open3d as o3d
import numpy as np

def pcd_to_pcd_fast(pcd_path, output_pcd_path, voxel_size=0.2):
    """
    最快方案：直接用Open3D降采样，然后从原文件匹配intensity
    """
    import time
    start = time.time()
    
    print("1. 读取并降采样点云...")
    # Open3D voxel_down_sample是C++实现，最快
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"原始点数: {len(pcd.points):,}")
    
    pcd_down = pcd.voxel_down_sample(voxel_size)
    down_points = np.asarray(pcd_down.points)
    print(f"降采样后: {len(down_points):,}")
    
    print("2. 从原文件匹配intensity...")
    # 找到降采样点对应的原始点（最近邻）
    from sklearn.neighbors import NearestNeighbors
    
    # 使用KDTree找最近邻（比scipy快）
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(np.asarray(pcd.points))
    _, indices = nn.kneighbors(down_points)
    indices = indices.flatten()
    
    print("3. 读取对应intensity...")
    # 快速读取指定行的intensity
    intensity = get_intensity_by_indices(pcd_path, indices)
    
    print("4. 保存...")
    save_pcd(output_pcd_path, down_points, intensity)
    
    print(f"总耗时: {time.time()-start:.2f}秒")
    return True

def get_intensity_by_indices(pcd_path, indices):
    """快速读取指定索引的intensity"""
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
    
    # 找数据起始行
    data_start = 0
    fields = []
    for i, line in enumerate(lines):
        if line.startswith('FIELDS'):
            fields = line.strip().split()[1:]
        elif line.startswith('DATA'):
            data_start = i + 1
            break
    
    # 找intensity位置
    intensity_idx = next((i for i, f in enumerate(fields) if f.lower() == 'intensity'), None)
    
    if intensity_idx is None:
        return np.zeros(len(indices), dtype=np.float32)
    
    # 批量读取
    intensity = np.zeros(len(indices), dtype=np.float32)
    line_cache = {}
    
    for i, idx in enumerate(indices):
        if idx in line_cache:
            intensity[i] = line_cache[idx]
        else:
            # 跳到对应行（文件太大时用seek更快）
            line = lines[data_start + idx]
            parts = line.split()
            val = float(parts[intensity_idx]) if len(parts) > intensity_idx else 0.0
            intensity[i] = val
            line_cache[idx] = val
    
    return intensity

def save_pcd(output_path, points, intensity):
    """保存PCD"""
    with open(output_path, 'w') as f:
        f.write("VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n")
        f.write("TYPE F F F F\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\nDATA ascii\n")
        
        # 用numpy批量写
        data = np.column_stack([points, intensity])
        np.savetxt(f, data, fmt='%.6f')

if __name__ == "__main__":
    pcd_to_pcd_fast(
        "/mnt/disk1/mapvln/pocket2/MT20260309-104919/MANIFOLD_MT20260309-104919-Cloud_Opt.pcd", # 留形App后处理优化后的重建点云
        "/mnt/disk1/mapvln/pocket2/MT20260309-104919/map/cloudGlobal.pcd", #用于fastlivo2定位地图
        voxel_size=0.2
    )