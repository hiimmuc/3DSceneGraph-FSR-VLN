import open3d as o3d
import numpy as np

def pcd_to_ply_fast(pcd_path, output_ply_path, voxel_size=0.2):
    """
    最快方案：直接用Open3D降采样，然后从原文件匹配intensity，保存为PLY格式
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
    
    print("4. 保存为PLY格式...")
    save_ply(output_ply_path, down_points, intensity)
    
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

def save_ply(output_path, points, intensity):
    """保存为PLY格式，包含xyz和intensity信息"""
    with open(output_path, 'w') as f:
        # PLY文件头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float intensity\n")
        f.write("end_header\n")
        
        # 写入数据
        for i in range(len(points)):
            f.write(f"{points[i][0]:.6f} {points[i][1]:.6f} {points[i][2]:.6f} {intensity[i]:.6f}\n")

def pcd_to_ply_simple(pcd_path, output_ply_path, voxel_size=0.2):
    """
    简化版本：使用Open3D直接保存为PLY（如果不需要精确匹配intensity）
    """
    import time
    start = time.time()
    
    print("读取点云...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"原始点数: {len(pcd.points):,}")
    
    print("降采样...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"降采样后: {len(pcd_down.points):,}")
    
    print("保存为PLY...")
    # 使用Open3D直接保存为PLY
    o3d.io.write_point_cloud(output_ply_path, pcd_down)
    
    print(f"总耗时: {time.time()-start:.2f}秒")
    return True

if __name__ == "__main__":
    # 使用精确匹配intensity的版本
    pcd_to_ply_fast(
        "/mnt/disk1/mapvln/pocket2/MT20260309-104919/MANIFOLD_MT20260309-104919-Cloud_Opt.pcd",
        "/mnt/disk1/mapvln/pocket2/MT20260309-104919/map/cloudGlobal.ply",
        voxel_size=0.02
    )
    
    # 或者使用简化版本（如果不需要精确匹配intensity）
    # pcd_to_ply_simple(
    #     "/mnt/disk1/mapvln/pocket2/MT20260121-105400/map/cloudGlobal.pcd", 
    #     "/mnt/disk1/mapvln/pocket2/MT20260121-105400/map/cloudGlobal_downsampled.ply",
    #     voxel_size=0.2
    # )