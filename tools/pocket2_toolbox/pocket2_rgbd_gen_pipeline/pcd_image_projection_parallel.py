#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCD点云相机投影工具

功能：
- 将3D点云数据投影到2D相机图像上
- 支持ASCII和二进制格式的PCD文件
- 自动查找相机外参文件
- 支持鱼眼相机模型和畸变校正
- 提供多种点云可视化模式

输入文件：
- PCD文件：包含3D点云数据
- 相机位姿文件：cam_X_pos.txt格式
- 目标图片：imgX_Y.jpg格式
- 外参文件：cam_in_ex(t).txt（自动查找）

作者：Assistant
版本：2.0
"""

import os
import re
import glob
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import platform
import cv2
import numpy as np
import threading
import subprocess
from scipy.spatial.transform import Rotation as R
import multiprocessing
from multiprocessing import shared_memory
import json
import time

# -------------------- Module-level helpers for multiprocessing --------------------
def _shm_create_from_array(arr: np.ndarray, name_prefix="pcd_shm"):
    """Create a shared memory block and write numpy array bytes into it. Returns metadata dict."""
    if arr is None:
        return None, None
    arr = np.asarray(arr)
    nbytes = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    # create a numpy view on the shared buffer and copy
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr[:]
    meta = {
        'name': shm.name,
        'shape': arr.shape,
        'dtype': str(arr.dtype)
    }
    # return both meta and the SharedMemory object so the creator can manage lifetime
    return meta, shm


def _shm_attach_array(meta):
    if meta is None:
        # return a (None, None) tuple so callers can safely unpack
        return None, None
    shm = shared_memory.SharedMemory(name=meta['name'])
    dtype = np.dtype(meta['dtype'])
    arr = np.ndarray(tuple(meta['shape']), dtype=dtype, buffer=shm.buf)
    return arr, shm


def _cleanup_shm(meta, shm):
    try:
        if shm is not None:
            shm.close()
            shm.unlink()
    except Exception:
        pass


# Lightweight, standalone implementations of the core functions used by workers.
def project_points_impl(points, intensities, T_w_cam, params, img_shape, cam_id):
    # Adapted from the class method; pure function (no self)
    points_homo = np.hstack([points, np.ones((len(points), 1))])
    T_cw = np.linalg.inv(np.array(T_w_cam))
    points_cam = (T_cw @ points_homo.T).T[:, :3]
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    depths = points_cam[:, 2]
    intensities = intensities[valid] if intensities is not None else None

    if len(points_cam) == 0:
        return np.array([]).reshape(3, 0), np.array([]), np.array([]).reshape(0,2), np.array([]), np.array([])

    # # Use the same intrinsics as the class's fallback (these may be overwritten by params)
    # fx = params.get('A11', params.get('fx', 294.6212))
    # fy = params.get('A22', params.get('fy', 294.8052))
    # cx = params.get('u0', params.get('cx', 329.3184))
    # cy = params.get('v0', params.get('cy', 270.3924))
    # skew = params.get('skew', 0.0)

    # # distortion coefficients (use params if present)
    # k2 = params.get('k2', 0.0)
    # k3 = params.get('k3', 0.0)
    # k4 = params.get('k4', 0.0)
    # k5 = params.get('k5', 0.0)
    # k6 = params.get('k6', 0.0)
    # k7 = params.get('k7', 0.0)
    # fx = 294.6212
    # fy = 294.8052
    # cx = 329.3184
    # cy = 270.3924
    image_root_dir = "/mnt/disk1/mapvln/pocket2/MT20260309-104919/image"
    K_dist = np.loadtxt(os.path.join(image_root_dir, f"rectify_{cam_id}_param.txt"))
    fx = K_dist[0, 0]
    fy = K_dist[1, 1]
    cx = K_dist[0, 2]
    cy = K_dist[1, 2]
    skew = 0.0
    
    # 畸变参数
    k2 = 0.0
    k3 = 0.0
    k4 = 0.0
    k5 = 0.0
    k6 = 0.0
    k7 = 0.0

    is_fisheye = ('k1' not in params) and abs(k2) > 1e-10

    if is_fisheye:
        norm = np.linalg.norm(points_cam, axis=1, keepdims=True)
        uv_norm = points_cam / norm
        r = np.sqrt(uv_norm[:, 0]**2 + uv_norm[:, 1]**2)
        theta = np.arccos(np.clip(uv_norm[:, 2], -1.0, 1.0))
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta3 * theta
        theta5 = theta4 * theta
        theta6 = theta5 * theta
        theta7 = theta6 * theta
        theta_d = theta + k2*theta2 + k3*theta3 + k4*theta4 + k5*theta5 + k6*theta6 + k7*theta7
        scaling = np.where(r > 1e-8, theta_d / r, 1.0)
        x_dist = uv_norm[:, 0] * scaling
        y_dist = uv_norm[:, 1] * scaling
        u = fx * x_dist + skew * y_dist + cx
        v = fy * y_dist + cy
    else:
        x_norm = points_cam[:, 0] / points_cam[:, 2]
        y_norm = points_cam[:, 1] / points_cam[:, 2]
        k1 = params.get('k1', 0.0)
        p1 = params.get('p1', 0.0)
        p2 = params.get('p2', 0.0)
        r2 = x_norm**2 + y_norm**2
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        x_dist = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
        y_dist = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
        u = fx * x_dist + skew * y_dist + cx
        v = fy * y_dist + cy

    h, w = img_shape[:2]
    valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depths_valid = depths[valid_mask]
    points_camera = points_cam[valid_mask].T
    points_image = np.concatenate([u_valid.reshape(-1,1), v_valid.reshape(-1,1), depths_valid.reshape(-1,1)], axis=1).T
    if intensities is not None:
        intensities_valid = intensities[valid_mask]
    else:
        intensities_valid = None
    return points_image, points_camera, np.column_stack([u_valid, v_valid]), depths_valid, intensities_valid


def whether_occluded_deoccfast_impl(uvs, img_input, image_scale, dilate_view_path=""):
    imgH, imgW = img_input.shape[:2]
    occlusion_flag = [False] * len(uvs)
    depth = np.zeros((imgH, imgW), dtype=np.float32)
    inv_depth = np.zeros((imgH, imgW), dtype=np.float32)
    min_depth = np.full((imgH, imgW), 1000.0, dtype=np.float32)
    fb = 20.0
    for k, (x, y, z) in enumerate(uvs):
        if z <= 0 or x + 0.5 < 0 or x + 0.5 >= imgW or y + 0.5 < 0 or y + 0.5 >= imgH:
            occlusion_flag[k] = True
            continue
        col = int(x + 0.5)
        row = int(y + 0.5)
        cur_depth = z
        if min_depth[row, col] > cur_depth:
            min_depth[row, col] = cur_depth
            depth[row, col] = cur_depth
            inv_depth[row, col] = fb / cur_depth
    kernel_size = max(1, 4 // image_scale)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    inv_dilate_depth = cv2.dilate(inv_depth, element, iterations=4)
    inv_dilate_depth_s16 = np.int16(inv_dilate_depth)
    cv2.filterSpeckles(inv_dilate_depth_s16, 0, 1000, 1)
    for k, (x, y, z) in enumerate(uvs):
        if z <= 0 or x + 0.5 < 0 or x + 0.5 >= imgW or y + 0.5 < 0 or y + 0.5 >= imgH:
            occlusion_flag[k] = True
            continue
        col = int(x + 0.5)
        row = int(y + 0.5)
        noise = inv_dilate_depth_s16[row, col]
        if noise == 0:
            occlusion_flag[k] = True
            continue
        pesudo_disp = inv_dilate_depth_s16[row, col]
        px_disp = fb / z
        if abs(px_disp - pesudo_disp) >= 2.0:
            occlusion_flag[k] = True
    return occlusion_flag


def generate_occ_depth_impl(points_image, points_camera, img_width, img_height, dilate_view_path, depth_output_path, visualization_output_path, depth_factor, img_input, image_scale, logger=None):
    uvs = np.vstack((points_image[0, :], points_image[1, :], points_camera[2, :])).T
    occlusion_flags = np.array(whether_occluded_deoccfast_impl(uvs, img_input, image_scale, dilate_view_path))
    valid_mask = (uvs[:, 2] > 0) & ~occlusion_flags
    valid_uvs = uvs[valid_mask]
    x = valid_uvs[:, 0].astype(int)
    y = valid_uvs[:, 1].astype(int)
    z = valid_uvs[:, 2]
    in_bounds_mask = (0 <= x) & (x < img_width) & (0 <= y) & (y < img_height)
    x, y, z = x[in_bounds_mask], y[in_bounds_mask], z[in_bounds_mask]
    depth_map = np.zeros((img_height, img_width), dtype=np.float32)
    depth_map[y, x] = z * depth_factor
    colors = None
    if valid_uvs.size > 0:
        # 构建 0-255 的单通道强度序列，传入 applyColorMap，然后保证输出为 (N,3)
        vals = np.uint8(np.clip((valid_uvs[:, 2] - valid_uvs[:, 2].min()) / max(1e-6, valid_uvs[:, 2].ptp()) * 255, 0, 255))
        vals = vals.reshape(-1, 1)  # (N,1)
        cmap = cv2.applyColorMap(vals, cv2.COLORMAP_JET)  # (N,1,3)
        # Flatten to (N,3)
        colors = cmap.reshape(-1, 3)
    for i, (col, row) in enumerate(zip(x, y)):
        color = colors[i] if colors is not None else (0,255,0)
        cv2.circle(img_input, (col, row), 1, (int(color[0]), int(color[1]), int(color[2])), -1)
    cv2.imwrite(visualization_output_path, img_input)
    cv2.imwrite(depth_output_path, depth_map.astype(np.uint16))
    if logger:
        logger(f"Saved depth map to: {depth_output_path}")
        logger(f"Saved visualization to: {visualization_output_path}")


def overlay_points_impl(image, pixel_coords, depths, intensities, alpha=1.0, color_mode='intensity'):
    result = image.copy()
    if len(pixel_coords) == 0:
        return result
    h, w = image.shape[:2]
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    pcd_img = np.zeros((h, w, 3), dtype=np.uint8)
    us = pixel_coords[:, 0].astype(np.int32)
    vs = pixel_coords[:, 1].astype(np.int32)
    valid_mask = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    us_valid = us[valid_mask]
    vs_valid = vs[valid_mask]
    depths_valid = depths[valid_mask]
    if len(us_valid) == 0:
        return result
    if color_mode == 'intensity' and intensities is not None:
        intensities_valid = intensities[valid_mask]
        valid_intensities = intensities_valid[np.isfinite(intensities_valid)]
        if len(valid_intensities) > 0 and (valid_intensities.max() - valid_intensities.min()) > 1e-6:
            i_min, i_max = valid_intensities.min(), valid_intensities.max()
            norm_i = np.clip((intensities_valid - i_min) / (i_max - i_min), 0.0, 1.0)
            colors_uint8 = (norm_i * 255).astype(np.uint8)
            colors = cv2.applyColorMap(colors_uint8.reshape(-1, 1), cv2.COLORMAP_JET)[:, 0, :]
        else:
            color_mode = 'depth'
    if color_mode == 'depth':
        d_min, d_max = depths_valid.min(), depths_valid.max()
        if d_max > d_min:
            norm_d = (depths_valid - d_min) / (d_max - d_min)
            colors_uint8 = (norm_d * 255).astype(np.uint8)
            colors = cv2.applyColorMap(colors_uint8.reshape(-1, 1), cv2.COLORMAP_JET)[:, 0, :]
        else:
            colors = np.full((len(us_valid), 3), [0, 255, 0], dtype=np.uint8)
    elif color_mode == 'green':
        colors = np.full((len(us_valid), 3), [0, 255, 0], dtype=np.uint8)
    sort_indices = np.argsort(-depths_valid)
    us_sorted = us_valid[sort_indices]
    vs_sorted = vs_valid[sort_indices]
    depths_sorted = depths_valid[sort_indices]
    colors_sorted = colors[sort_indices]
    point_size = 1
    for i in range(len(us_sorted)):
        u, v, depth, color = us_sorted[i], vs_sorted[i], depths_sorted[i], colors_sorted[i]
        if depth < depth_map[v, u]:
            depth_map[v, u] = depth
            pcd_img[v, u] = color
    mask = np.any(pcd_img > 0, axis=2)
    result[mask] = cv2.addWeighted(pcd_img[mask], alpha, result[mask], 1 - alpha, 0)
    return result


def _process_frame_worker(args_json):
    """Top-level worker for ProcessPoolExecutor. Receives a JSON-serializable dict as string."""
    args = json.loads(args_json)
    # attach shared arrays
    pcd_arr, pcd_shm = _shm_attach_array(args.get('pcd_meta'))
    intens_arr, intens_shm = _shm_attach_array(args.get('int_meta'))
    try:
        points = np.array(pcd_arr)
        intensities = np.array(intens_arr) if intens_arr is not None else None
        T_w_cam = np.array(args['T_w_cam'])
        params = args['params']
        img_path = args['img_path']
        frame_id = args['frame_id']
        cam_id = args['cam_id']
        depth_dir = args['depth_dir']
        depth_vis_dir = args['depth_vis_dir']
        output_dir = args['output_dir']
        alpha = args.get('alpha', 1.0)
        color_mode = args.get('color_mode', 'intensity')

        image = cv2.imread(img_path)
        if image is None:
            return {'frame_id': frame_id, 'success': False, 'msg': f'无法加载图片: {img_path}'}

        points_image, points_camera, pixel_coords, depths, intensities_valid = project_points_impl(points, intensities, T_w_cam, params, image.shape, cam_id)

        dilate_view_path = os.path.join(depth_vis_dir, f"{frame_id:06d}_dilate.png")
        depth_output_path = os.path.join(depth_dir, f"{frame_id:06d}.png")
        visualization_output_path = os.path.join(depth_vis_dir, f"{frame_id:06d}_overlay.png")

        generate_occ_depth_impl(points_image, points_camera, image.shape[1], image.shape[0], dilate_view_path, depth_output_path, visualization_output_path, depth_factor=1000, img_input=image, image_scale=1)

        result = overlay_points_impl(image, pixel_coords, depths, intensities_valid, alpha=alpha, color_mode=color_mode)
        output_name = f"{cam_id}_frame{frame_id}_overlay.jpg"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, result)
        return {'frame_id': frame_id, 'success': True, 'msg': f'保存: {output_name}'}
    finally:
        try:
            if 'pcd_shm' in locals() and pcd_shm is not None:
                pcd_shm.close()
        except Exception:
            pass
        try:
            if 'intens_shm' in locals() and intens_shm is not None:
                intens_shm.close()
        except Exception:
            pass


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory adaptive downsampling disabled")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    print("Warning: Open3D not available")


# image: 1600x1296
# stereo_image: 1920x1080

# 常量定义
# DEFAULT_OUTPUT_DIR = "/mnt/disk1/mapvln/pocket2/MT20251210-114035/projection_results"
# DEFAULT_OUTPUT_DIR = "/mnt/disk1/mapvln/pocket2/MT20251211-195452/projection_results"
DEFAULT_OUTPUT_DIR = "/mnt/disk1/mapvln/pocket2/MT20260309-104919/projection_results"
DEFAULT_ALPHA = 1.0
SUPPORTED_PCD_FORMATS = [("PCD文件", "*.pcd *.bin")]
SUPPORTED_POSE_FORMATS = [("位姿文件", "cam_*_pos.txt *.txt")]
SUPPORTED_IMAGE_FORMATS = [("图片", "*.jpg *.jpeg *.JPEG *.png")]
CALIB_FILE_PATTERNS = [
    "MT*/image/cam_in_ex.txt", "MT*/image/cam_in_ext.txt",
]


class SimplePCDProjection:
    """PCD点云相机投影工具主类"""
    
    def __init__(self, root):
        """初始化GUI应用程序"""
        self.root = root
        self.root.title("PCD点云相机投影工具")
        # 动态根据屏幕大小和 DPI 设定窗口尺寸及字体缩放，避免跨平台字体/大小问题
        try:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            # DPI: 像素/英寸
            screen_dpi = max(72.0, float(self.root.winfo_fpixels('1i')))
            # 相对缩放因子（以 96 DPI 为基准）
            scale = screen_dpi / 96.0
            # 限制缩放范围，避免过大或过小
            scale = max(0.8, min(scale, 2.5))

            win_w = int(screen_w * 0.6)
            win_h = int(screen_h * 0.7)
            # 设置窗口大小并居中
            x = (screen_w - win_w) // 2
            y = (screen_h - win_h) // 2
            self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

            # 配置可伸缩字体（修改 Tk 默认字体，影响大部分 ttk 小部件）
            default_font = tkfont.nametofont('TkDefaultFont')
            default_size = max(9, int(default_font.cget('size') * scale))
            default_font.configure(size=default_size)

            text_font = tkfont.nametofont('TkTextFont')
            text_font.configure(size=max(10, int(text_font.cget('size') * scale)))

            fixed_font = tkfont.nametofont('TkFixedFont')
            fixed_font.configure(size=max(10, int(fixed_font.cget('size') * scale)))

            # 标题字体，使用系统默认族以避免在某些平台找不到 SimHei
            self.title_font = tkfont.Font(family=default_font.cget('family'),
                                          size=max(14, int(16 * scale)), weight='bold')

            # 应用 ttk 主题并设置通用样式
            try:
                style = ttk.Style()
                # 在不同平台选择较现代的主题
                if platform.system() == 'Windows':
                    style.theme_use('vista')
                else:
                    style.theme_use('clam')
                style.configure('.', font=default_font)
                style.configure('TButton', padding=6)
            except Exception:
                pass
        except Exception:
            # 任何异常都回退到默认窗口大小
            self.root.geometry("800x700")
        
        # 初始化数据存储
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.pcd_points = None
        self.pcd_intensities = None
        self.camera_poses = {}  # {cam_id: {frame_id: {'T_w_cam': T, 'timestamp': ts}}}
        self.camera_params = {}  # {cam_id: params}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建GUI界面
        self.create_widgets()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 标题
        # ttk.Label(self.root, text="PCD点云相机投影工具", 
        #          font=("SimHei", 16, "bold")).pack(pady=10)
        ttk.Label(self.root, text="PCD点云相机投影工具", font=self.title_font).pack(pady=10)
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文件选择
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        # PCD文件
        self.create_file_row(file_frame, "PCD文件:", "pcd_entry", self.select_pcd)
        
        # 位姿文件
        self.create_file_row(file_frame, "相机位姿文件:", "pose_entry", self.select_pose)
        
        # 目标图片
        self.create_file_row(file_frame, "目标图片:", "img_entry", self.select_image)
        
        # 外参文件
        self.create_file_row(file_frame, "内外参文件:", "calib_entry", self.select_calib)
        
        # 输出目录
        row = ttk.Frame(file_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="输出目录:", width=15).pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(row, width=55)
        self.output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.output_entry.insert(0, self.output_dir)
        ttk.Button(row, text="选择", command=self.select_output, width=10).pack(side=tk.LEFT)
        
        # 文件信息
        info_frame = ttk.LabelFrame(main_frame, text="文件信息", padding="10")
        info_frame.pack(fill=tk.X, pady=10)
        
        self.info_text = tk.Text(info_frame, height=4, width=70, wrap=tk.WORD, state='disabled')
        self.info_text.pack(fill=tk.X)
        
        # 显示参数
        param_frame = ttk.LabelFrame(main_frame, text="显示参数", padding="10")
        param_frame.pack(fill=tk.X, pady=10)
        
        # 透明度
        alpha_row = ttk.Frame(param_frame)
        alpha_row.pack(fill=tk.X, pady=5)
        ttk.Label(alpha_row, text="点云透明度:", width=15).pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=DEFAULT_ALPHA)
        alpha_scale = ttk.Scale(alpha_row, from_=0.0, to=1.0, 
                               variable=self.alpha_var, orient=tk.HORIZONTAL)
        alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.alpha_label = ttk.Label(alpha_row, text="0.33", width=5)
        self.alpha_label.pack(side=tk.LEFT)
        alpha_scale.config(command=lambda v: self.alpha_label.config(text=f"{float(v):.2f}"))
        
        # 颜色模式
        color_row = ttk.Frame(param_frame)
        color_row.pack(fill=tk.X, pady=5)
        ttk.Label(color_row, text="点云颜色:", width=15).pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value="intensity")
        ttk.Radiobutton(color_row, text="强度", variable=self.color_var, 
                       value="intensity").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(color_row, text="深度", variable=self.color_var, 
                       value="depth").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(color_row, text="绿色", variable=self.color_var, 
                       value="green").pack(side=tk.LEFT, padx=5)
        
        # 按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.process_btn = ttk.Button(btn_frame, text="开始投影", command=self.start_process)
        self.process_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(btn_frame, text="打开结果", command=self.open_output).pack(side=tk.LEFT, padx=5)
        
        # 进度
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(progress_frame, textvariable=self.status_var).pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 日志
        log_frame = ttk.Frame(progress_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=12, width=70, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_file_row(self, parent, label, entry_name, command):
        """创建文件选择行"""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=label, width=15).pack(side=tk.LEFT)
        entry = ttk.Entry(row, width=55)
        entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        setattr(self, entry_name, entry)
        ttk.Button(row, text="选择", command=command, width=10).pack(side=tk.LEFT)
    
    def log(self, msg):
        """记录日志"""
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_info(self, msg):
        """更新信息显示"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, msg)
        self.info_text.config(state='disabled')
    
    def select_pcd(self):
        """选择PCD文件"""
        path = filedialog.askopenfilename(title="选择PCD文件", filetypes=SUPPORTED_PCD_FORMATS)
        if path:
            self.pcd_entry.delete(0, tk.END)
            self.pcd_entry.insert(0, path)
            # self.load_pcd(path)
            self.pcd_points, self.pcd_intensities = self.load_pcd_with_intensity(path)
    
    def select_pose(self):
        """选择位姿文件"""
        path = filedialog.askopenfilename(title="选择相机位姿文件", filetypes=SUPPORTED_POSE_FORMATS)
        if path:
            self.pose_entry.delete(0, tk.END)
            self.pose_entry.insert(0, path)
            self.load_pose(path)
    
    def select_image(self):
        """选择目标图片"""
        path = filedialog.askopenfilename(title="选择目标图片", filetypes=SUPPORTED_IMAGE_FORMATS)
        if path:
            self.img_entry.delete(0, tk.END)
            self.img_entry.insert(0, path)
            self.extract_image_info(path)
    
    def select_calib(self):
        """选择外参文件（可选）"""
        path = filedialog.askopenfilename(
            title="选择外参文件（可选）", 
            filetypes=[("外参文件", "cam_in_ex*.txt *.txt"), ("所有文件", "*.*")]
        )
        if path:
            self.calib_entry.delete(0, tk.END)
            self.calib_entry.insert(0, path)
            # 如果已有位姿数据，立即加载外参
            if self.camera_poses:
                cam_id = list(self.camera_poses.keys())[0]
                self.load_calib(path, cam_id)
                self.log(f"  ✓ 手动加载外参文件: {os.path.basename(path)}")
    
    def select_output(self):
        """选择输出目录"""
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)
            self.output_dir = path
    
    def open_output(self):
        """打开输出目录"""
        if os.path.exists(self.output_dir):
            subprocess.run(["xdg-open", self.output_dir])
    def load_pcd_with_intensity(self, pcd_file):
        with open(pcd_file, 'rb') as f:
            header_lines = []
            for _ in range(20):
                line = f.readline()
                try:
                    line_str = line.decode('ascii').strip()
                    header_lines.append(line_str)
                    if line_str.startswith('DATA'):
                        data_format = line_str.split()[1] if len(line_str.split()) > 1 else 'ascii'
                        break
                except:
                    break
        
        if 'binary' in data_format.lower():
            self.log(f"  Detected binary format PCD")
            points, intensities = self.load_pcd_binary(pcd_file)
        else:
            self.log(f"  Detected ASCII format PCD")
            points, intensities = self.load_pcd_ascii_xyzi(pcd_file)
        
        # points, intensities = self.adaptive_downsample(points, intensities, pcd_file)

        points, intensities = self.voxel_downsample(points, None, voxel_size=0.01)
        
        return points, intensities

    def voxel_downsample(self, points, intensities, voxel_size=0.002):
        """
        快速的 voxel 下采样（基于 NumPy 的 voxel-grid 分组），避免对每个下采样点进行 KDTree 查询。

        实现思路：对点坐标除以 voxel_size 向下取整得到体素索引（3D 整数索引），
        使用 np.unique(..., axis=0, return_inverse=True) 将点分组，然后对每个体素内的点取均值作为代表点，
        同时对强度也取平均值。如果 intensities 为 None，则返回 None。

        这个方法对大规模点云在速度与内存上通常比 Open3D+KDTree 更优，且完全用向量化操作完成。
        """
        pts = np.asarray(points)
        if pts.size == 0:
            return pts, intensities

        # 防御性检查
        if voxel_size <= 0:
            self.log("  ⚠ voxel_size must be > 0, skipping downsample")
            return pts, intensities

        # 计算体素索引（向下取整），支持负坐标
        voxel_idx = np.floor(pts / float(voxel_size)).astype(np.int64)

        # 找到每个唯一体素，并得到每个点属于哪个体素（inverse）以及每个体素的点数
        uniq_voxels, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)

        # 按体素累加坐标并求平均
        sums = np.zeros((uniq_voxels.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inverse, pts)
        down_points = (sums / counts[:, None]).astype(np.float32)

        # 如果有强度，则对强度也做体素内平均
        if intensities is not None:
            ints = np.asarray(intensities)
            sums_i = np.zeros((uniq_voxels.shape[0],), dtype=np.float64)
            np.add.at(sums_i, inverse, ints)
            down_intensities = (sums_i / counts).astype(np.float32)
        else:
            down_intensities = None

        ratio = len(down_points) / len(pts) if len(pts) > 0 else 0
        self.log(f"  ✓ Voxel downsampling (numpy) complete: {len(pts)} → {len(down_points)} points ({ratio:.2%} remaining)")

        return down_points, down_intensities
    
    def adaptive_downsample(self, points, intensities, pcd_file):
        if not PSUTIL_AVAILABLE:
            return points, intensities
        
        mem = psutil.virtual_memory()
        available_mem_bytes = mem.available
        available_mem_gb = available_mem_bytes / (1024**3)
        
        num_points = len(points)
        estimated_mem_bytes = num_points * (3 * 8 + 8)
        estimated_mem_gb = estimated_mem_bytes / (1024**3)
        
        self.log(f"  Memory detection: Available={available_mem_gb:.2f}GB, Point cloud estimated={estimated_mem_gb:.2f}GB ({num_points} points)")
        
        if estimated_mem_bytes > available_mem_bytes:
            target_mem_bytes = available_mem_bytes * 0.9
            target_points = int(target_mem_bytes / 32)
            downsample_ratio = target_points / num_points
            
            self.log(f"  ⚠ Insufficient memory! Downsampling needed...")
            self.log(f"  Target point count: {target_points} (downsample ratio: {downsample_ratio:.2%})")
            
            np.random.seed(42)
            random_indices = np.random.choice(num_points, target_points, replace=False)
            random_indices = np.sort(random_indices)
            
            points_downsampled = points[random_indices]
            if intensities is not None:
                intensities_downsampled = intensities[random_indices]
            else:
                intensities_downsampled = None
            
            final_mem_gb = (len(points_downsampled) * 32) / (1024**3)
            self.log(f"  ✓ Downsampling complete: {num_points} → {len(points_downsampled)} points")
            self.log(f"  ✓ Estimated memory: {estimated_mem_gb:.2f}GB → {final_mem_gb:.2f}GB")
            
            return points_downsampled, intensities_downsampled
        else:
            self.log(f"  ✓ Sufficient memory, no downsampling needed")
            return points, intensities
    
    def load_pcd_binary(self, pcd_file):
        with open(pcd_file, 'rb') as f:
            fields = []
            sizes = []
            types = []
            counts = []
            num_points = 0
            header_end = 0
            data_format = 'binary'
            
            while True:
                pos = f.tell()
                line = f.readline()
                try:
                    line_str = line.decode('ascii').strip()
                except:
                    break
                
                if line_str.startswith('FIELDS'):
                    fields = line_str.split()[1:]
                elif line_str.startswith('SIZE'):
                    sizes = [int(x) for x in line_str.split()[1:]]
                elif line_str.startswith('TYPE'):
                    types = line_str.split()[1:]
                elif line_str.startswith('COUNT'):
                    counts = [int(x) for x in line_str.split()[1:]]
                elif line_str.startswith('POINTS'):
                    num_points = int(line_str.split()[1])
                elif line_str.startswith('DATA'):
                    data_format = line_str.split()[1] if len(line_str.split()) > 1 else 'binary'
                    header_end = f.tell()
                    break
            
            x_idx, y_idx, z_idx, intensity_idx = -1, -1, -1, -1
            for i, field in enumerate(fields):
                field_lower = field.lower()
                if field_lower == 'x':
                    x_idx = i
                elif field_lower == 'y':
                    y_idx = i
                elif field_lower == 'z':
                    z_idx = i
                elif field_lower in ['intensity', 'i']:
                    intensity_idx = i
            
            if x_idx < 0 or y_idx < 0 or z_idx < 0:
                raise Exception(f"Cannot recognize XYZ fields, FIELDS: {fields}")
            
            self.log(f"  ✓ Detected XYZ fields: x@{x_idx}, y@{y_idx}, z@{z_idx}")
            if intensity_idx >= 0:
                self.log(f"  ✓ Detected Intensity field: {fields[intensity_idx]}@{intensity_idx}")
            else:
                self.log(f"  ⚠ Intensity field not found")
            
            field_offsets = []
            offset = 0
            for i, (size, count) in enumerate(zip(sizes, counts)):
                field_offsets.append(offset)
                offset += size * count
            point_size = offset
            
            self.log(f"  Bytes per point: {point_size}, Total points: {num_points}")
            self.log(f"  Data format: {data_format}")
            
            self.log(f"  Field offset mapping:")
            self.log(f"    x[{x_idx}] -> offset {field_offsets[x_idx]}")
            self.log(f"    y[{y_idx}] -> offset {field_offsets[y_idx]}")
            self.log(f"    z[{z_idx}] -> offset {field_offsets[z_idx]}")
            if intensity_idx >= 0:
                self.log(f"    intensity[{intensity_idx}] -> offset {field_offsets[intensity_idx]}")
            
            f.seek(header_end)
            compressed_data = f.read()
            
            if data_format == 'binary_compressed':
                self.log(f"  Detected compressed format，Starting decompression...")
                self.log(f"  Compressed data size: {len(compressed_data)} bytes")
                try:
                    data = self.decompress_lzf(compressed_data, num_points * point_size)
                    self.log(f"  Decompressed data size: {len(data)} bytes")
                except Exception as e:
                    raise Exception(f"LZF decompression failed: {str(e)}")
                
                self.log(f"  ⚠ binary_compressed uses column-major storage, parsing by fields...")
                
                field_arrays = {}
                offset = 0
                for i, (field, size, count) in enumerate(zip(fields, sizes, counts)):
                    field_size = size * count
                    field_bytes = num_points * field_size
                    
                    if offset + field_bytes > len(data):
                        raise Exception(f"Data incomplete: Field {field} needs {field_bytes} bytes, but only {len(data)-offset} bytes available")
                    
                    field_data = data[offset:offset + field_bytes]
                    
                    if types[i] == 'F':
                        if size == 4:
                            field_arrays[field] = np.frombuffer(field_data, dtype=np.float32)
                        elif size == 8:
                            field_arrays[field] = np.frombuffer(field_data, dtype=np.float64)
                    elif types[i] == 'U':
                        field_arrays[field] = np.frombuffer(field_data, dtype=f'<u{size}')
                    elif types[i] == 'I':
                        field_arrays[field] = np.frombuffer(field_data, dtype=f'<i{size}')
                    else:
                        if size == 4:
                            field_arrays[field] = np.frombuffer(field_data, dtype=np.float32)
                        elif size == 8:
                            field_arrays[field] = np.frombuffer(field_data, dtype=np.float64)
                    
                    self.log(f"    Field '{field}': {len(field_arrays[field])} values, range [{field_arrays[field].min():.2f}, {field_arrays[field].max():.2f}]")
                    offset += field_bytes
                
                x_field = fields[x_idx]
                y_field = fields[y_idx]
                z_field = fields[z_idx]
                
                points_array = np.column_stack([
                    field_arrays[x_field],
                    field_arrays[y_field],
                    field_arrays[z_field]
                ]).astype(np.float32)
                
                if intensity_idx >= 0:
                    intensity_field = fields[intensity_idx]
                    intensities_array = field_arrays[intensity_field].astype(np.float32)
                else:
                    intensities_array = None
                
            else:
                data = compressed_data
                if len(data) < num_points * point_size:
                    raise Exception(f"Data incomplete: Expected {num_points * point_size} bytes, Actual {len(data)} bytes")
                
                self.log(f"  Using numpy batch parsing (row-major storage)...")
                
                dtype_list = []
                
                if point_size == 64 and len(fields) == 6:
                    self.log(f"  Building dtype with 64-byte padding format...")
                    for i, (field, ftype, size) in enumerate(zip(fields, types, sizes)):
                        if ftype == 'F':
                            dtype_list.append((field, '<f4'))
                        elif ftype == 'U':
                            dtype_list.append((field, f'<u{size}'))
                        elif ftype == 'I':
                            dtype_list.append((field, f'<i{size}'))
                        
                        if i < 3:
                            dtype_list.append((f'_pad{i}', 'V12'))
                        elif i == 5:
                            dtype_list.append(('_pad_end', 'V4'))
                else:
                    for i, (field, ftype, size) in enumerate(zip(fields, types, sizes)):
                        if ftype == 'F':
                            if size == 4:
                                dtype_list.append((field, '<f4'))
                            elif size == 8:
                                dtype_list.append((field, '<f8'))
                        elif ftype == 'U':
                            dtype_list.append((field, f'<u{size}'))
                        elif ftype == 'I':
                            dtype_list.append((field, f'<i{size}'))
                        else:
                            if size == 4:
                                dtype_list.append((field, '<f4'))
                            elif size == 8:
                                dtype_list.append((field, '<f8'))
                
                self.log(f"  Building dtype: {dtype_list}")
                dt = np.dtype(dtype_list)
                
                points_data = np.frombuffer(data, dtype=dt)
                
                x_field = fields[x_idx]
                y_field = fields[y_idx]
                z_field = fields[z_idx]
                
                points_array = np.column_stack([
                    points_data[x_field],
                    points_data[y_field],
                    points_data[z_field]
                ]).astype(np.float32)
                
                if intensity_idx >= 0:
                    intensity_field = fields[intensity_idx]
                    intensities_array = points_data[intensity_field].astype(np.float32)
                else:
                    intensities_array = None
            
            self.log(f"  Verifying parsing result (first 3 points):")
            for i in range(min(3, len(points_array))):
                intensity_val = intensities_array[i] if intensities_array is not None else 'N/A'
                self.log(f"    Point{i}: x={points_array[i,0]:.2f}, y={points_array[i,1]:.2f}, z={points_array[i,2]:.2f}, intensity={intensity_val}")
            
            self.log(f"  Parsing complete, verifying range:")
            self.log(f"    X: [{points_array[:, 0].min():.2f}, {points_array[:, 0].max():.2f}]")
            self.log(f"    Y: [{points_array[:, 1].min():.2f}, {points_array[:, 1].max():.2f}]")
            self.log(f"    Z: [{points_array[:, 2].min():.2f}, {points_array[:, 2].max():.2f}]")
            if intensities_array is not None:
                self.log(f"    Intensity: [{intensities_array.min():.2f}, {intensities_array.max():.2f}]")
            
            return points_array, intensities_array
    
    def decompress_lzf(self, compressed_data, expected_size):
        if len(compressed_data) < 8:
            raise Exception(f"Compressed data too short: {len(compressed_data)} bytes")
        
        compressed_size = struct.unpack('<I', compressed_data[0:4])[0]
        uncompressed_size = struct.unpack('<I', compressed_data[4:8])[0]
        
        self.log(f"  Compression block info: Compressed={compressed_size}, Uncompressed={uncompressed_size}")
        
        if uncompressed_size != expected_size:
            self.log(f"  Warning: Uncompressed size mismatch ({uncompressed_size} vs {expected_size})")
        
        try:
            import lzf
            self.log(f"  Using lzf library for decompression...")
            chunk_data = compressed_data[8:8+compressed_size]
            result = lzf.decompress(chunk_data, uncompressed_size)
            return result
        except ImportError:
            self.log(f"  lzf library not installed, using manual implementation...")
        except Exception as e:
            self.log(f"  lzf library decompression failed: {e}, trying manual implementation...")
        
        chunk_data = compressed_data[8:8+compressed_size]
        decompressed = self.lzf_decompress_chunk(chunk_data, uncompressed_size)
        
        if len(decompressed) != uncompressed_size:
            raise Exception(f"Decompression size mismatch: Expected {uncompressed_size}, Actual {len(decompressed)}")
        
        return decompressed
    
    def lzf_decompress_chunk(self, data, uncompressed_size):
        output = bytearray()
        pos = 0
        
        while pos < len(data):
            ctrl = data[pos]
            pos += 1
            
            if ctrl < 32:
                length = ctrl + 1
                if pos + length > len(data):
                    break
                output.extend(data[pos:pos+length])
                pos += length
            else:
                length = ctrl >> 5
                if length == 7:
                    length += data[pos]
                    pos += 1
                
                if pos >= len(data):
                    break
                    
                offset = ((ctrl & 0x1f) << 8) | data[pos]
                pos += 1
                offset += 1
                length += 2
                
                if offset > len(output):
                    break
                
                ref_pos = len(output) - offset
                for _ in range(length):
                    if ref_pos < len(output):
                        output.append(output[ref_pos])
                        ref_pos += 1
        
        return bytes(output)
    
    def load_pcd_ascii_xyzi(self, pcd_file):
        points = []
        intensities = []
        fields = []
        x_idx, y_idx, z_idx, intensity_idx = -1, -1, -1, -1
        
        with open(pcd_file, 'r') as f:
            data_section = False
            for line in f:
                line = line.strip()
                
                if line.startswith('FIELDS'):
                    fields = line.split()[1:]
                    
                    for i, field in enumerate(fields):
                        field_lower = field.lower()
                        if field_lower == 'x':
                            x_idx = i
                        elif field_lower == 'y':
                            y_idx = i
                        elif field_lower == 'z':
                            z_idx = i
                        elif field_lower in ['intensity', 'i']:
                            intensity_idx = i
                    
                    if x_idx >= 0 and y_idx >= 0 and z_idx >= 0:
                        self.log(f"  ✓ Detected XYZ fields: x@{x_idx}, y@{y_idx}, z@{z_idx}")
                    else:
                        raise Exception(f"Cannot recognize XYZ fields, FIELDS: {fields}")
                    
                    if intensity_idx >= 0:
                        self.log(f"  ✓ Detected Intensity field: {fields[intensity_idx]}@{intensity_idx}")
                    else:
                        self.log(f"  ⚠ Intensity field not found")
                    
                    continue
                
                if line.startswith('DATA'):
                    data_section = True
                    continue
                    
                if data_section and line:
                    parts = line.split()
                    if len(parts) > max(x_idx, y_idx, z_idx):
                        try:
                            x = float(parts[x_idx])
                            y = float(parts[y_idx])
                            z = float(parts[z_idx])
                            points.append([x, y, z])
                            
                            if intensity_idx >= 0 and len(parts) > intensity_idx:
                                intensities.append(float(parts[intensity_idx]))
                        except:
                            continue
        
        points_array = np.array(points, dtype=np.float32)
        intensities_array = np.array(intensities, dtype=np.float32) if intensities else None
        
        if intensities_array is not None and len(intensities_array) != len(points_array):
            self.log(f"  ⚠ Intensity count mismatch, ignoring intensity field")
            intensities_array = None
        
        return points_array, intensities_array
    
    def load_pcd_ascii(self, pcd_file):
        points, _ = self.load_pcd_ascii_xyzi(pcd_file)
        return points
        
    def load_pcd(self, path):
        """加载PCD文件"""
        self.log(f"加载PCD: {os.path.basename(path)}")
        try:
            points, intensities = self.parse_pcd(path)
            
            # 添加点云统计信息
            self.log(f"  原始点云: {len(points)} 个点")
            if len(points) > 0:
                self.log(f"  点云范围: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                self.log(f"           Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
                self.log(f"           Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
                if intensities is not None:
                    self.log(f"  强度范围: [{intensities.min():.2f}, {intensities.max():.2f}]")
            
            self.pcd_points = points
            self.pcd_intensities = intensities
            self.log(f"  ✓ 加载完成")
        except Exception as e:
            self.log(f"  ✗ 失败: {e}")
            messagebox.showerror("错误", f"加载PCD失败:\n{e}")
    
    def parse_pcd(self, path):
        """解析PCD文件（支持ASCII和二进制格式）"""
        points, intensities = [], []
        
        # 首先读取头部信息
        header_info = {}
        data_format = 'ascii'
        header_lines = []
        
        with open(path, 'rb') as f:
            # 读取头部
            while True:
                line = f.readline()
                if not line:
                    break
                
                try:
                    line_str = line.decode('utf-8').strip()
                except:
                    break
                
                header_lines.append(line_str)
                
                if line_str.startswith('FIELDS'):
                    header_info['fields'] = line_str.split()[1:]
                elif line_str.startswith('SIZE'):
                    header_info['sizes'] = [int(x) for x in line_str.split()[1:]]
                elif line_str.startswith('TYPE'):
                    header_info['types'] = line_str.split()[1:]
                elif line_str.startswith('COUNT'):
                    header_info['counts'] = [int(x) for x in line_str.split()[1:]]
                elif line_str.startswith('WIDTH'):
                    header_info['width'] = int(line_str.split()[1])
                elif line_str.startswith('HEIGHT'):
                    header_info['height'] = int(line_str.split()[1])
                elif line_str.startswith('POINTS'):
                    header_info['points'] = int(line_str.split()[1])
                elif line_str.startswith('DATA'):
                    data_format = line_str.split()[1].lower()
                    break
            
            # 解析数据
            if data_format == 'ascii':
                return self._parse_ascii_pcd(f, header_info)
            else:
                return self._parse_binary_pcd(f, header_info)
    
    def _parse_ascii_pcd(self, f, header_info):
        """解析ASCII格式PCD数据"""
        points, intensities = [], []
        fields = header_info.get('fields', [])
        
        # 找到字段索引
        x_idx = fields.index('x') if 'x' in fields else -1
        y_idx = fields.index('y') if 'y' in fields else -1
        z_idx = fields.index('z') if 'z' in fields else -1
        i_idx = -1
        for i, field in enumerate(fields):
            if field.lower() in ['intensity', 'i']:
                i_idx = i
                break
        
        # 读取剩余数据
        for line in f:
            try:
                line_str = line.decode('utf-8').strip()
                if line_str:
                    parts = line_str.split()
                    if len(parts) > max(x_idx, y_idx, z_idx):
                        x, y, z = float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])
                        points.append([x, y, z])
                        intensities.append(float(parts[i_idx]) if i_idx >= 0 and len(parts) > i_idx else 1.0)
            except:
                continue
        
        return np.array(points), np.array(intensities)
    
    def _parse_binary_pcd(self, f, header_info):
        """解析二进制格式PCD数据"""
        
        fields = header_info.get('fields', [])
        sizes = header_info.get('sizes', [])
        types = header_info.get('types', [])
        counts = header_info.get('counts', [])
        num_points = header_info.get('points', 0)
        
        if num_points == 0:
            return np.array([]), np.array([])
        
        # 找到字段索引
        x_idx = fields.index('x') if 'x' in fields else -1
        y_idx = fields.index('y') if 'y' in fields else -1
        z_idx = fields.index('z') if 'z' in fields else -1
        i_idx = -1
        for i, field in enumerate(fields):
            if field.lower() in ['intensity', 'i']:
                i_idx = i
                break
        
        if x_idx == -1 or y_idx == -1 or z_idx == -1:
            return np.array([]), np.array([])
        
        # 计算每个点的字节大小
        point_size = sum(sizes[i] * counts[i] for i in range(len(fields)))
        
        # 构建struct格式字符串
        fmt_chars = []
        for i, (size, type_char, count) in enumerate(zip(sizes, types, counts)):
            if type_char == 'F':
                fmt_chars.extend(['f'] * count)
            elif type_char == 'U':
                if size == 1:
                    fmt_chars.extend(['B'] * count)
                elif size == 2:
                    fmt_chars.extend(['H'] * count)
                elif size == 4:
                    fmt_chars.extend(['I'] * count)
            elif type_char == 'I':
                if size == 1:
                    fmt_chars.extend(['b'] * count)
                elif size == 2:
                    fmt_chars.extend(['h'] * count)
                elif size == 4:
                    fmt_chars.extend(['i'] * count)
        
        fmt = '<' + ''.join(fmt_chars)  # 小端格式
        
        points, intensities = [], []
        
        try:
            for _ in range(num_points):
                data = f.read(point_size)
                if len(data) < point_size:
                    break
                
                values = struct.unpack(fmt, data)
                
                x, y, z = values[x_idx], values[y_idx], values[z_idx]
                points.append([x, y, z])
                
                intensity = values[i_idx] if i_idx >= 0 and i_idx < len(values) else 1.0
                intensities.append(intensity)
                
        except Exception as e:
            self.log(f"  ⚠ 二进制解析错误: {e}")
        
        return np.array(points), np.array(intensities)
    
    def load_pose(self, path):
        """加载位姿文件"""
        self.log(f"加载位姿: {os.path.basename(path)}")
        try:
            # 提取相机ID
            match = re.match(r'(cam_\d+)_pos\.txt', os.path.basename(path))
            if not match:
                raise Exception("文件名格式错误，应为 cam_X_pos.txt")
            
            cam_id = match.group(1)
            self.camera_poses = {cam_id: {}}
            
            # 读取位姿
            count = 0
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 9:
                        frame_id = int(parts[0])
                        timestamp = float(parts[1])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        qw, qx, qy, qz = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                        
                        # 构建变换矩阵（用户已计算好的相机位姿）
                        rot = R.from_quat([qx, qy, qz, qw])
                        T_w_cam = np.eye(4)
                        T_w_cam[:3, :3] = rot.as_matrix()
                        T_w_cam[:3, 3] = [x, y, z]
                        
                        self.camera_poses[cam_id][frame_id] = {
                            'T_w_cam': T_w_cam,  # 用户已计算好的相机位姿
                            'timestamp': timestamp
                        }
                        count += 1
            
            self.log(f"  ✓ {cam_id}: {count} 个位姿")
            
            # 自动查找并加载外参文件
            self._find_and_load_calib(path, cam_id)
            
            # 更新信息
            min_id = min(self.camera_poses[cam_id].keys())
            max_id = max(self.camera_poses[cam_id].keys())
            self.update_info(f"相机: {cam_id}\n位姿数量: {count}\n帧范围: {min_id} - {max_id}")
            
        except Exception as e:
            self.log(f"  ✗ 失败: {e}")
            messagebox.showerror("错误", f"加载位姿失败:\n{e}")
    
    def _find_and_load_calib(self, pose_path, cam_id):
        """查找并加载外参文件"""
        # 优先使用手动选择的外参文件
        if hasattr(self, 'calib_entry') and self.calib_entry.get().strip():
            manual_calib_path = self.calib_entry.get().strip()
            if os.path.exists(manual_calib_path):
                self.load_calib(manual_calib_path, cam_id)
                self.log(f"  ✓ 使用手动选择的外参文件")
                return True
            else:
                self.log(f"  ⚠ 手动选择的外参文件不存在: {manual_calib_path}")
        
        # 如果没有手动选择，则自动查找
        folder = os.path.dirname(pose_path)
        parent_folder = os.path.dirname(folder)
        
        # 构建搜索路径
        search_paths = []
        for pattern in CALIB_FILE_PATTERNS:
            if pattern.startswith("MT*"):
                search_paths.append(os.path.join(parent_folder, pattern))
            else:
                search_paths.append(os.path.join(folder, pattern))
        
        # 查找外参文件
        for pattern in search_paths:
            matches = glob.glob(pattern)
            if matches:
                calib_path = matches[0]
                self.load_calib(calib_path, cam_id)
                self.log(f"  ✓ 自动找到外参文件: {os.path.basename(calib_path)}")
                return True
        
        self.log(f"  ⚠ 未找到外参文件，投影时将使用默认参数")
        return False
    
    def load_calib(self, path, cam_id):
        """加载外参文件"""
        self.log(f"加载外参: {os.path.basename(path)}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 初始化Tcl存储
            if not hasattr(self, 'camera_tcl'):
                self.camera_tcl = {}
            
            # 解析Tcl矩阵（参考V2版本）
            cam_num = cam_id.split("_")[1]
            tcl_pattern = rf'Tcl_{cam_num}:\s*\[([^\]]+)\]'
            tcl_match = re.search(tcl_pattern, content, re.MULTILINE | re.DOTALL)
            
            if tcl_match:
                values_str = tcl_match.group(1)
                values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)
                
                if len(values) >= 16:
                    Tcl = np.array([float(v) for v in values[:16]]).reshape(4, 4)
                    self.camera_tcl[cam_id] = Tcl
                    self.log(f"  ✓ 找到{cam_id}的Tcl外参矩阵")
                else:
                    self.log(f"  ⚠ Tcl_{cam_num}数据不完整，只有{len(values)}个值")
            else:
                self.log(f"  ⚠ 未找到Tcl_{cam_num}外参")
            
            # 解析相机参数
            pattern = f'cam_{cam_num}:\\s*\\n((?:.*\\n)*?)(?=cam_\\d+:|Tcl_\\d+:|$)'
            self.log(f"  查找相机参数: {cam_id} (cam_{cam_num})")
            match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            
            if match:
                params = {}
                section = match.group(1)
                
                param_patterns = {
                    'image_width': r'image_width:\s*(\d+)',
                    'image_height': r'image_height:\s*(\d+)',
                    'k2': r'k2:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'k3': r'k3:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'k4': r'k4:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'k5': r'k5:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'k6': r'k6:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'k7': r'k7:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'A11': r'A11:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'A12': r'A12:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'A22': r'A22:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'u0': r'u0:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                    'v0': r'v0:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
                }
                
                for name, pattern in param_patterns.items():
                    m = re.search(pattern, section)
                    if m:
                        params[name] = int(m.group(1)) if name in ['image_width', 'image_height'] else float(m.group(1))
                
                self.camera_params[cam_id] = params
                self.log(f"  ✓ {cam_id}: {params.get('image_width', 'N/A')}x{params.get('image_height', 'N/A')}")
            else:
                # 尝试查找所有可用的相机ID
                available_cams = re.findall(r'cam_(\d+):', content)
                self.log(f"  ✗ 未找到 cam_{cam_num} 的参数")
                self.log(f"  可用相机: {available_cams}")
                
        except Exception as e:
            self.log(f"  ✗ 外参加载失败: {e}")
    
    def extract_image_info(self, path):
        """从图片文件名提取信息并显示对应帧的位姿"""
        basename = os.path.basename(path)
        match = re.match(r'img(\d+)_(\d+)', os.path.splitext(basename)[0])
        if match:
            cam_idx, frame_id = match.group(1), int(match.group(2))
            cam_id = f'cam_{cam_idx}'
            
            self.log(f"图片: {basename} → {cam_id}, 帧{frame_id}")
            
            # 查找并打印对应帧的位姿信息到日志中
            self._get_frame_pose_info(cam_id, frame_id)
            
            # 更新信息面板（不包含位姿信息）
            info = self.info_text.get(1.0, tk.END).strip()
            info += f"\n\n图片: {basename}\n相机索引: {cam_idx}\n帧ID: {frame_id}"
            self.update_info(info)
    
    def _get_frame_pose_info(self, cam_id, frame_id):
        """获取并打印指定帧的位姿信息到日志中"""
        if cam_id not in self.camera_poses:
            self.log(f"  ⚠ 未找到{cam_id}的位姿数据")
            return
        
        if frame_id not in self.camera_poses[cam_id]:
            self.log(f"  ⚠ 未找到{cam_id}帧{frame_id}的位姿")
            # 显示可用的帧范围
            available_frames = list(self.camera_poses[cam_id].keys())
            if available_frames:
                min_frame = min(available_frames)
                max_frame = max(available_frames)
                self.log(f"  可用帧范围: {min_frame} - {max_frame}")
            return
        
        # 获取位姿数据
        pose_data = self.camera_poses[cam_id][frame_id]
        T_w_cam = pose_data['T_w_cam']
        timestamp = pose_data['timestamp']
        
        # 提取位置和旋转
        position = T_w_cam[:3, 3]
        rotation_matrix = T_w_cam[:3, :3]
        
        # 转换为四元数
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # [x, y, z, w]
        
        # 在日志中打印详细信息
        self.log(f"  ✓ 找到{cam_id}帧{frame_id}的位姿:")
        self.log(f"    时间戳: {timestamp:.6f}")
        self.log(f"    位置: [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]")
        self.log(f"    四元数: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
    
    def _use_default_camera_params(self):
        """使用默认相机参数"""
        # 从当前加载的位姿中获取相机ID
        if not self.camera_poses:
            return
        
        cam_id = list(self.camera_poses.keys())[0]  # 使用第一个相机ID
        
        # 默认相机参数（适用于1920x1080分辨率）
        default_params = {
            'image_width': 1920,
            'image_height': 1080,
            'A11': 1000.0,  # fx - 焦距
            'A22': 1000.0,  # fy - 焦距
            'A12': 0.0,     # skew - 倾斜
            'u0': 960.0,    # cx - 主点x
            'v0': 540.0,    # cy - 主点y
            'k2': 0.0,      # 畸变系数
            'k3': 0.0,
            'k4': 0.0,
            'k5': 0.0,
            'k6': 0.0,
            'k7': 0.0
        }
        
        self.camera_params[cam_id] = default_params
        self.log(f"  ⚠ 使用默认相机参数: {cam_id}")
        self.log(f"    分辨率: {default_params['image_width']}x{default_params['image_height']}")
        self.log(f"    焦距: fx={default_params['A11']}, fy={default_params['A22']}")
        self.log(f"    主点: cx={default_params['u0']}, cy={default_params['v0']}")
    
    def _handle_missing_camera_params(self):
        """处理缺失相机参数的情况"""
        # 创建参数选择对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("相机参数设置")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明文字
        ttk.Label(frame, text="未找到相机外参文件，请选择处理方式:", 
                 font=("Arial", 10, "bold")).pack(pady=(0, 10))
        
        # 选项变量
        option_var = tk.StringVar(value="default")
        
        # 选项1：使用默认参数
        ttk.Radiobutton(frame, text="使用默认参数（快速，精度较低）", 
                       variable=option_var, value="default").pack(anchor=tk.W, pady=5)
        
        # 选项2：手动输入参数
        ttk.Radiobutton(frame, text="手动输入相机参数（精度较高）", 
                       variable=option_var, value="manual").pack(anchor=tk.W, pady=5)
        
        # 选项3：取消
        ttk.Radiobutton(frame, text="取消投影，先加载外参文件", 
                       variable=option_var, value="cancel").pack(anchor=tk.W, pady=5)
        
        # 参数输入框（初始隐藏）
        param_frame = ttk.LabelFrame(frame, text="相机参数", padding="10")
        
        # 创建参数输入控件
        param_vars = {}
        param_labels = {
            'fx': '焦距 fx:', 'fy': '焦距 fy:', 'cx': '主点 cx:', 'cy': '主点 cy:',
            'k2': '畸变 k2:', 'k3': '畸变 k3:', 'k4': '畸变 k4:'
        }
        
        for i, (key, label) in enumerate(param_labels.items()):
            row = ttk.Frame(param_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            var = tk.StringVar(value="1000" if key in ['fx', 'fy'] else 
                              "960" if key == 'cx' else "540" if key == 'cy' else "0")
            param_vars[key] = var
            ttk.Entry(row, textvariable=var, width=15).pack(side=tk.LEFT, padx=5)
        
        def on_option_change():
            if option_var.get() == "manual":
                param_frame.pack(fill=tk.X, pady=10)
            else:
                param_frame.pack_forget()
        
        # 绑定选项变化事件
        for widget in frame.winfo_children():
            if isinstance(widget, ttk.Radiobutton):
                widget.configure(command=on_option_change)
        
        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        result = {'action': None}
        
        def on_confirm():
            action = option_var.get()
            if action == "default":
                self._use_default_camera_params()
                result['action'] = 'proceed'
            elif action == "manual":
                try:
                    # 验证并应用手动输入的参数
                    manual_params = {}
                    for key, var in param_vars.items():
                        manual_params[key] = float(var.get())
                    self._use_manual_camera_params(manual_params)
                    result['action'] = 'proceed'
                except ValueError:
                    messagebox.showerror("错误", "参数格式不正确，请输入数字")
                    return
            else:  # cancel
                result['action'] = 'cancel'
            
            dialog.destroy()
        
        ttk.Button(btn_frame, text="确定", command=on_confirm).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="取消", command=lambda: (setattr(result, 'action', 'cancel'), dialog.destroy())).pack(side=tk.RIGHT)
        
        # 等待对话框关闭
        dialog.wait_window()
        
        # 根据结果决定是否继续
        if result['action'] != 'proceed':
            self.process_btn.config(state='normal')
            self.progress_bar.stop()
            return False
        
        return True
    
    def _use_manual_camera_params(self, manual_params):
        """使用手动输入的相机参数"""
        if not self.camera_poses:
            return
        
        cam_id = list(self.camera_poses.keys())[0]
        
        # 构建完整的参数字典
        params = {
            'image_width': 1920,
            'image_height': 1080,
            'A11': manual_params.get('fx', 1000.0),
            'A22': manual_params.get('fy', 1000.0),
            'A12': 0.0,
            'u0': manual_params.get('cx', 960.0),
            'v0': manual_params.get('cy', 540.0),
            'k2': manual_params.get('k2', 0.0),
            'k3': manual_params.get('k3', 0.0),
            'k4': manual_params.get('k4', 0.0),
            'k5': 0.0,
            'k6': 0.0,
            'k7': 0.0
        }
        
        self.camera_params[cam_id] = params
        self.log(f"  ✓ 使用手动输入的相机参数: {cam_id}")
        self.log(f"    焦距: fx={params['A11']}, fy={params['A22']}")
        self.log(f"    主点: cx={params['u0']}, cy={params['v0']}")
        self.log(f"    畸变: k2={params['k2']}, k3={params['k3']}, k4={params['k4']}")
    
    def start_process(self):
        """开始处理"""
        # 检查输入
        if not self.pcd_entry.get() or self.pcd_points is None:
            messagebox.showerror("错误", "请先选择PCD文件")
            return
        
        if len(self.pcd_points) == 0:
            messagebox.showerror("错误", "PCD文件中没有有效的点云数据")
            return
        
        if not self.pose_entry.get() or not self.camera_poses:
            messagebox.showerror("错误", "请先选择位姿文件")
            return
        
        if not self.img_entry.get():
            messagebox.showerror("错误", "请先选择目标图片")
            return
        
        if not self.camera_params:
            # 如果没有加载外参文件，提供选择
            if not self._handle_missing_camera_params():
                return  # 用户取消了操作
        
        self.process_btn.config(state='disabled')
        self.progress_bar.start()
        
        thread = threading.Thread(target=self.process, daemon=True)
        thread.start()

    def depth2color(self, points, min_depth=0.2, max_depth=1000.0):
        """
        Convert depth values to a color map representation.
        
        :param points: List of (x, y, z) tuples representing 3D points.
        :param min_depth: Minimum depth value for normalization.
        :param max_depth: Maximum depth value for normalization.
        :return: Color map as a numpy array.
        """
        if points.size == 0:  # 检查输入是否为空
            return np.zeros((1, 0, 3), dtype=np.uint8)  # 返回空的颜色映射
        N = len(points)
        dist_gray = np.zeros((1, N), dtype=np.uint8)
        min_depth = np.min(points[:, 2])
        max_depth = np.max(points[:, 2])
        for i in range(N):
            dist = points[i][2]  # Assuming points is a list of (x, y, z) tuples
            dist = np.clip(dist, min_depth, max_depth)
            ratio = (dist - min_depth) / (max_depth - min_depth)  # 归一化到 [0, 1]
            ratio = np.sqrt(ratio)  # 使用平方根增强近处的颜色变化
            dist_gray[0, i] = int(ratio * 255)
        # print(f"dist_gray: {dist_gray}")
        dist_color = cv2.applyColorMap(dist_gray, cv2.COLORMAP_JET)
        # print(f"dist_color: {dist_color}")
        return dist_color

    def whether_occluded_deoccfast(self, uvs, img_input, image_scale, dilate_view_path=""):
        """
        判断点是否被遮挡的函数。
        
        参数:
            uvs (list of tuple): 包含 (x, y, z) 的点列表。
            img_input (numpy.ndarray): 输入图像，用于获取图像尺寸。
            image_scale (int): 图像缩放比例，用于调整膨胀操作的核大小。
            dilate_view_path (str): 可选，保存膨胀结果的路径。
        
        返回:
            list of bool: 每个点是否被遮挡的标志。
        """
        imgH, imgW = img_input.shape[:2]
        occlusion_flag = [False] * len(uvs)
        depth = np.zeros((imgH, imgW), dtype=np.float32)
        inv_depth = np.zeros((imgH, imgW), dtype=np.float32)
        min_depth = np.full((imgH, imgW), 1000.0, dtype=np.float32)
        fb = 20.0

        # 多对一取最近的点（前景）
        for k, (x, y, z) in enumerate(uvs):
            if z <= 0 or x + 0.5 < 0 or x + 0.5 >= imgW or y + 0.5 < 0 or y + 0.5 >= imgH:
                occlusion_flag[k] = True
                continue

            col = int(x + 0.5)
            row = int(y + 0.5)
            # col= int(x)
            # row= int(y)
            cur_depth = z
            if min_depth[row, col] > cur_depth:  # 当前点距离更近
                min_depth[row, col] = cur_depth
                depth[row, col] = cur_depth
                inv_depth[row, col] = fb / cur_depth

        # 膨胀操作
        kernel_size = max(1, 4 // image_scale)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        inv_dilate_depth = cv2.dilate(inv_depth, element, iterations=4)
        inv_dilate_depth_s16 = np.int16(inv_dilate_depth)
        # inv_dilate_depth_s16 = np.uint8(np.clip(inv_dilate_depth_s16, 0, 255))
        cv2.filterSpeckles(inv_dilate_depth_s16, 0, 1000, 1)
        # 保存膨胀结果（可选）
        # if dilate_view_path:
        #     # inv_depth_u8 = np.uint8(inv_depth * 8)
        #     # inv_dilate_depth_u8 = np.uint8(inv_dilate_depth * 8)
        #     inv_dilate_depth_u8 = inv_dilate_depth.astype(np.uint8)
        #     inv_depth_u8 = inv_depth.astype(np.uint8)
        #     cv2.filterSpeckles(inv_dilate_depth_u8, 0, 2000, 2)

        #     disp_vis = cv2.applyColorMap(inv_depth_u8*8, cv2.COLORMAP_JET)
        #     disp_dilate_vis = cv2.applyColorMap(inv_dilate_depth_u8*8, cv2.COLORMAP_JET)
        #     result = np.hstack((disp_vis, disp_dilate_vis))
        #     cv2.imwrite(dilate_view_path, result)
        for k, (x, y, z) in enumerate(uvs):
            if z <= 0 or x + 0.5 < 0 or x + 0.5 >= imgW or y + 0.5 < 0 or y + 0.5 >= imgH:
                occlusion_flag[k] = True
                continue

            col = int(x + 0.5)
            row = int(y + 0.5)        
            # col= int(x)
            # row= int(y)
            # noise = inv_dilate_depth_s16[row, col]
            noise = inv_dilate_depth_s16[row, col]
            if noise == 0:
                # print(f"Warning: noise is zero at index {k}, setting occlusion_flag to True")
                occlusion_flag[k] = True
                continue
            pesudo_disp = inv_dilate_depth_s16[row, col]
            px_disp = fb / z
            if abs(px_disp - pesudo_disp) >= 2.0:
                occlusion_flag[k] = True

        return occlusion_flag


    def generate_occ_depth(self, points_image, points_camera, img_width, img_height, dilate_view_path, depth_output_path, visualization_output_path, depth_factor, img_input, image_scale):
        """
        生成深度图，并对点云进行去遮挡处理。
        :param points_image: 投影到图像上的点云坐标 (numpy array)
        :param points_camera: 相机坐标系下的点云坐标 (numpy array)
        :param img_width: 图像宽度
        :param img_height: 图像高度
        :param depth_factor: 深度缩放因子
        :param img_input: 输入图像，用于遮挡判断
        :param image_scale: 图像缩放比例
        """
        # 将点云转换为 (x, y, z) 格式
        uvs = np.vstack((points_image[0, :], points_image[1, :], points_camera[2, :])).T
        # print(f"uvs size: {uvs.shape}")

        # 调用 whether_occluded_deoccfast 进行去遮挡
        occlusion_flags = np.array(self.whether_occluded_deoccfast(uvs, img_input, image_scale, dilate_view_path))

        # 筛选未被遮挡的点
        valid_mask = (uvs[:, 2] > 0) & ~occlusion_flags
        # valid_mask= (uvs[:, 2] > 0)
        valid_uvs = uvs[valid_mask]

        # 提取有效点的坐标和深度
        x = valid_uvs[:, 0].astype(int)
        y = valid_uvs[:, 1].astype(int)
        z = valid_uvs[:, 2]

        # 筛选在图像范围内的点
        in_bounds_mask = (0 <= x) & (x < img_width) & (0 <= y) & (y < img_height)
        x, y, z = x[in_bounds_mask], y[in_bounds_mask], z[in_bounds_mask]

        # 初始化深度图
        depth_map = np.zeros((img_height, img_width), dtype=np.float32)

        # 填充深度图
        depth_map[y, x] = z * depth_factor

        # 生成颜色映射
        colors = self.depth2color(valid_uvs, min_depth=0.2, max_depth=100.0)
        # 可视化点云颜色
        for i, (col, row) in enumerate(zip(x, y)):
            color = colors[0][i]
            cv2.circle(img_input, (col, row), 1, (int(color[0]), int(color[1]), int(color[2])), -1)
        # 保存结果
        cv2.imwrite(visualization_output_path, img_input)
        cv2.imwrite(depth_output_path, depth_map.astype(np.uint16))
        self.log(f"Saved depth map to: {depth_output_path}")
        self.log(f"Saved visualization to: {visualization_output_path}")



    def process(self):
        """处理流程"""
        try:
            self.log("="*60)
            self.log("开始投影...")
            self.status_var.set("处理中...")
            start_time = time.time()
            
            img_path = self.img_entry.get()
            basename = os.path.basename(img_path)
            
            # # 提取信息
            # match = re.match(r'img(\d+)_(\d+)', os.path.splitext(basename)[0])
            # if not match:
            #     raise Exception("无法从文件名提取信息")
            
            # cam_id = f'cam_{match.group(1)}'
            img_dir = os.path.dirname(img_path)
            cam_id = os.path.basename(img_dir).replace('rectify_', '')  # cam_0           
            img_path = None
            
            # 构建待处理帧列表（按文件名解析 frame_id），并行执行每帧处理（使用多进程 + 共享内存）
            files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPEG"))])
            frames = []
            for frame_id in range(0, len(files)):
                img_path = os.path.join(img_dir, files[frame_id])

                # 检查数据
                if cam_id not in self.camera_poses:
                    raise Exception(f"未找到{cam_id}的位姿")
                
                if frame_id not in self.camera_poses[cam_id]:
                    raise Exception(f"{cam_id}的帧{frame_id}无位姿")
                
                if cam_id not in self.camera_params:
                    raise Exception(f"未找到{cam_id}的参数")
                frames.append((frame_id, img_path))

            if not frames:
                raise Exception("未找到要处理的图像")

            # 保持原有的 depth / depth_vis 保存路径不变
            self.depth_dir = f"/mnt/disk1/mapvln/pocket2/MT20260309-104919/image/rectify_depth_parallel/{cam_id}"
            self.depth_vis_dir = f"/mnt/disk1/mapvln/pocket2/MT20260309-104919/image/rectify_depth_vis_parallel/{cam_id}"
            self.image_root_dir = "/mnt/disk1/mapvln/pocket2/MT20260309-104919/image"
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.depth_vis_dir, exist_ok=True)

            # 准备共享内存（点云可能很大，避免复制）
            pcd_meta, pcd_shm = _shm_create_from_array(self.pcd_points)
            int_meta, int_shm = _shm_create_from_array(self.pcd_intensities) if self.pcd_intensities is not None else (None, None)

            # 限制 OpenCV 内部线程以避免嵌套线程导致卡死
            try:
                cv2.setNumThreads(1)
            except Exception:
                pass

            max_workers = min(16, (os.cpu_count() or 4))
            chunk_size = max(1, max_workers * 2)
            self.log(f"使用进程数: {max_workers}，待处理帧数: {len(frames)}，分批大小: {chunk_size}")

            # 创建进程池并分批提交任务
            all_args = []
            # debug helper: compare one-frame serial projection (class method) with worker inputs
            debug_logged = False
            for frame_id, img_path_local in frames:
                # 检查位姿与参数
                if cam_id not in self.camera_poses or frame_id not in self.camera_poses[cam_id] or cam_id not in self.camera_params:
                    self.log(f"跳过 帧{frame_id}: 缺少位姿或参数")
                    continue

                T_w_cam = self.camera_poses[cam_id][frame_id]['T_w_cam'].tolist()
                params = self.camera_params[cam_id]

                args = {
                    'pcd_meta': pcd_meta,
                    'int_meta': int_meta,
                    'T_w_cam': T_w_cam,
                    'params': params,
                    'img_path': img_path_local,
                    'frame_id': frame_id,
                    'cam_id': cam_id,
                    'depth_dir': self.depth_dir,
                    'depth_vis_dir': self.depth_vis_dir,
                    'output_dir': self.output_dir,
                    'alpha': float(self.alpha_var.get()),
                    'color_mode': self.color_var.get()
                }
                all_args.append(args)
                # 在第一个有效帧上输出本地（串行）投影样本，便于与并行结果比对
                if not debug_logged:
                    try:
                        img = cv2.imread(img_path_local)
                        if img is not None:
                            pts_img, pts_cam, pix_coords, deps, ints = self.project_points(self.pcd_points, self.pcd_intensities, np.array(T_w_cam), params, img.shape, cam_id)
                            # 打印前 10 个像素坐标样本
                            if isinstance(pix_coords, np.ndarray) and len(pix_coords) > 0:
                                sample = pix_coords[:10]
                                self.log(f"DEBUG sample pix coords (first 10) for frame {frame_id}: {sample.tolist()}")
                            else:
                                self.log(f"DEBUG: no pixels projected for frame {frame_id} in main process")
                    except Exception as e:
                        self.log(f"DEBUG projection sample failed: {e}")
                    debug_logged = True

            results = []
            import json
            with multiprocessing.get_context('spawn').Pool(processes=max_workers) as pool:
                # 使用 map 分批执行以限制同时在跑的进程数量并获得稳健性
                for i in range(0, len(all_args), chunk_size):
                    batch = all_args[i:i+chunk_size]
                    json_args = [json.dumps(a) for a in batch]
                    for res in pool.imap_unordered(_process_frame_worker, json_args):
                        frame_id = res.get('frame_id')
                        success = res.get('success')
                        msg = res.get('msg')
                        if success:
                            self.log(f"  ✓ 帧{frame_id}: {msg}")
                        else:
                            self.log(f"  ✗ 帧{frame_id}: {msg}")
                        results.append((frame_id, success, msg))

            # 清理共享内存（确保在进程池退出后再做）
            try:
                if 'pcd_shm' in locals() and pcd_shm is not None:
                    try:
                        pcd_shm.close()
                    except Exception:
                        pass
                    try:
                        pcd_shm.unlink()
                    except Exception:
                        pass
                if 'int_shm' in locals() and int_shm is not None:
                    try:
                        int_shm.close()
                    except Exception:
                        pass
                    try:
                        int_shm.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

            successes = sum(1 for _, s, _ in results if s)
            self.log(f"处理完成: 成功 {successes}/{len(frames)} 帧")
            self.status_var.set("完成")
            end_time = time.time()
            self.log(f"处理时间: {end_time - start_time:.2f} 秒")
            self.log(f"每个帧处理时间: {(end_time - start_time) / len(frames):.4f} 秒")

                
            
            
        except Exception as e:
            self.log(f"✗ 失败: {e}")
            self.status_var.set("失败")
            messagebox.showerror("错误", f"处理失败:\n{e}")
        finally:
            self.progress_bar.stop()
            self.process_btn.config(state='normal')
    
        # 投影点云
    def project_points_new(points, rotation, translation, intrinsics, img_width, img_height):
        # 将点云转换到相机坐标系
        points_camera = np.dot(rotation, points.T) + translation.reshape(3, 1)

        # 剔除 z <= 0 的点（相机后方的点）
        valid_mask = points_camera[2, :] > 0
        points_camera = points_camera[:, valid_mask] # 3XN

        # 投影到图像平面
        points_image = np.dot(intrinsics, points_camera / points_camera[2, :]) # 3xN

        # 剔除超出图像范围的点
        x, y = points_image[0, :], points_image[1, :]
        valid_mask = (x >= 0) & (x < img_width) & (y >= 0) & (y < img_height)
        points_camera = points_camera[:, valid_mask] # 3XN, xyz
        points_image = points_image[:, valid_mask] # 3xN, uvd

        return points_image, points_camera
    
    def project_points(self, points, intensities, T_w_cam, params, img_shape, cam_id):
        """投影点云到图像（直接使用相机位姿）"""
        self.log(f"  使用直接相机位姿投影（无需IMU→LiDAR→Camera变换）")
        
        # 转换到相机坐标系（直接使用相机位姿）
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        
        # 直接计算：T_cw = inv(T_w_cam)
        T_cw = np.linalg.inv(T_w_cam)
        points_cam = (T_cw @ points_homo.T).T[:, :3] # Nx3
        
        # 深度过滤
        self.log(f"  投影前点云数量: {len(points_cam)}")
        valid = points_cam[:, 2] > 0
        points_cam = points_cam[valid]
        depths = points_cam[:, 2]

        intensities = intensities[valid] if intensities is not None else None
        self.log(f"  深度过滤后点云数量: {len(points_cam)}")
        
        if len(points_cam) == 0:
            return np.array([]).reshape(0, 2), np.array([]), np.array([])
        
        # 相机内参
        # fx = 293.8016
        # fy = 293.8876
        # cx = 315.0696
        # cy = 251.5424
#         K_dist:  [[294.6212   0.     329.3184]
#  [  0.     294.8052 270.3924]
#  [  0.       0.       1.    ]]
        fx = 294.6212
        fy = 294.8052
        cx = 329.3184
        cy = 270.3924
        skew = 0.0
        
        # 畸变参数
        k2 = 0.0
        k3 = 0.0
        k4 = 0.0
        k5 = 0.0
        k6 = 0.0
        k7 = 0.0
        
        # 判断是否为鱼眼模型
        is_fisheye = 'k1' not in params and abs(k2) > 1e-10
        
        if is_fisheye:
            self.log(f"  使用鱼眼畸变模型")
            # 鱼眼投影
            # 归一化
            norm = np.linalg.norm(points_cam, axis=1, keepdims=True)
            uv_norm = points_cam / norm
            
            # 球面坐标
            r = np.sqrt(uv_norm[:, 0]**2 + uv_norm[:, 1]**2)
            theta = np.arccos(np.clip(uv_norm[:, 2], -1.0, 1.0))
            
            # 畸变（修复为与V2版本一致的多项式模型）
            theta2 = theta * theta
            theta3 = theta2 * theta
            theta4 = theta3 * theta
            theta5 = theta4 * theta
            theta6 = theta5 * theta
            theta7 = theta6 * theta
            
            theta_d = theta + k2*theta2 + k3*theta3 + k4*theta4 + k5*theta5 + k6*theta6 + k7*theta7
            
            scaling = np.where(r > 1e-8, theta_d / r, 1.0)
            x_dist = uv_norm[:, 0] * scaling
            y_dist = uv_norm[:, 1] * scaling
            
            # 像素坐标
            u = fx * x_dist + skew * y_dist + cx
            v = fy * y_dist + cy
        else:
            self.log(f"  使用针孔畸变模型")
            # 标准针孔模型投影
            x_norm = points_cam[:, 0] / points_cam[:, 2]
            y_norm = points_cam[:, 1] / points_cam[:, 2]
            
            # 标准径向畸变
            k1 = params.get('k1', 0.0)
            p1 = params.get('p1', 0.0)
            p2 = params.get('p2', 0.0)
            
            r2 = x_norm**2 + y_norm**2
            radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
            
            # 切向畸变
            x_dist = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
            y_dist = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
            
            # 像素坐标
            u = fx * x_dist + skew * y_dist + cx
            v = fy * y_dist + cy
        
        # 过滤
        h, w = img_shape[:2]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        self.log(f"  边界过滤后点云数量: {np.sum(valid)}")
        
        # 确保返回的数组维度正确
        u_valid = u[valid] # Nx1
        v_valid = v[valid] # Nx1
        depths_valid = depths[valid] # Nx1
        # import pdb; pdb.set_trace()
        points_camera = points_cam[valid].T # 3XN, xyz
        points_image = np.concatenate([u_valid.reshape(-1,1), v_valid.reshape(-1,1), depths_valid.reshape(-1,1)], axis=1).T  #3XN
        
        if len(u_valid) == 0:
            return points_image, points_camera, np.array([]).reshape(0, 2), np.array([]), np.array([])
        
        # 处理强度数组
        if intensities is not None:
            intensities_valid = intensities[valid]
        else:
            intensities_valid = None
        
        # 确保u_valid和v_valid都是1维数组
        u_valid = np.asarray(u_valid).flatten()
        v_valid = np.asarray(v_valid).flatten()
        
        return points_image, points_camera, np.column_stack([u_valid, v_valid]), depths_valid, intensities_valid
    
    def project_points_bk(self, points, intensities, T_w_cam, params, img_shape, cam_id):
        """投影点云到图像（直接使用相机位姿）"""
        self.log(f"  使用直接相机位姿投影（无需IMU→LiDAR→Camera变换）")
        
        # 转换到相机坐标系（直接使用相机位姿）
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        
        # 直接计算：T_cw = inv(T_w_cam)
        T_cw = np.linalg.inv(T_w_cam)
        points_cam = (T_cw @ points_homo.T).T[:, :3] # Nx3
        
        # 深度过滤
        self.log(f"  投影前点云数量: {len(points_cam)}")
        valid = points_cam[:, 2] > 0
        points_cam = points_cam[valid]
        depths = points_cam[:, 2]

        intensities = intensities[valid] if intensities is not None else None
        self.log(f"  深度过滤后点云数量: {len(points_cam)}")
        
        if len(points_cam) == 0:
            return np.array([]).reshape(0, 2), np.array([]), np.array([])
        
        # 相机内参
        fx = params.get('A11', 1000)
        fy = params.get('A22', 1000)
        cx = params.get('u0', img_shape[1]/2)
        cy = params.get('v0', img_shape[0]/2)
        skew = params.get('A12', 0)
        
        # 畸变参数
        k2 = params.get('k2', 0.0)
        k3 = params.get('k3', 0.0)
        k4 = params.get('k4', 0.0)
        k5 = params.get('k5', 0.0)
        k6 = params.get('k6', 0.0)
        k7 = params.get('k7', 0.0)
        
        # 判断是否为鱼眼模型
        is_fisheye = 'k1' not in params and abs(k2) > 1e-10
        
        if is_fisheye:
            self.log(f"  使用鱼眼畸变模型")
            # 鱼眼投影
            # 归一化
            norm = np.linalg.norm(points_cam, axis=1, keepdims=True)
            uv_norm = points_cam / norm
            
            # 球面坐标
            r = np.sqrt(uv_norm[:, 0]**2 + uv_norm[:, 1]**2)
            theta = np.arccos(np.clip(uv_norm[:, 2], -1.0, 1.0))
            
            # 畸变（修复为与V2版本一致的多项式模型）
            theta2 = theta * theta
            theta3 = theta2 * theta
            theta4 = theta3 * theta
            theta5 = theta4 * theta
            theta6 = theta5 * theta
            theta7 = theta6 * theta
            
            theta_d = theta + k2*theta2 + k3*theta3 + k4*theta4 + k5*theta5 + k6*theta6 + k7*theta7
            
            scaling = np.where(r > 1e-8, theta_d / r, 1.0)
            x_dist = uv_norm[:, 0] * scaling
            y_dist = uv_norm[:, 1] * scaling
            
            # 像素坐标
            u = fx * x_dist + skew * y_dist + cx
            v = fy * y_dist + cy
        else:
            self.log(f"  使用针孔畸变模型")
            # 标准针孔模型投影
            x_norm = points_cam[:, 0] / points_cam[:, 2]
            y_norm = points_cam[:, 1] / points_cam[:, 2]
            
            # 标准径向畸变
            k1 = params.get('k1', 0.0)
            p1 = params.get('p1', 0.0)
            p2 = params.get('p2', 0.0)
            
            r2 = x_norm**2 + y_norm**2
            radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
            
            # 切向畸变
            x_dist = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
            y_dist = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
            
            # 像素坐标
            u = fx * x_dist + skew * y_dist + cx
            v = fy * y_dist + cy
        
        # 过滤
        h, w = img_shape[:2]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        self.log(f"  边界过滤后点云数量: {np.sum(valid)}")
        
        # 确保返回的数组维度正确
        u_valid = u[valid] # Nx1
        v_valid = v[valid] # Nx1
        depths_valid = depths[valid] # Nx1
        # import pdb; pdb.set_trace()
        points_camera = points_cam[valid].T # 3XN, xyz
        points_image = np.concatenate([u_valid.reshape(-1,1), v_valid.reshape(-1,1), depths_valid.reshape(-1,1)], axis=1).T  #3XN
        
        if len(u_valid) == 0:
            return points_image, points_camera, np.array([]).reshape(0, 2), np.array([]), np.array([])
        
        # 处理强度数组
        if intensities is not None:
            intensities_valid = intensities[valid]
        else:
            intensities_valid = None
        
        # 确保u_valid和v_valid都是1维数组
        u_valid = np.asarray(u_valid).flatten()
        v_valid = np.asarray(v_valid).flatten()
        
        return points_image, points_camera, np.column_stack([u_valid, v_valid]), depths_valid, intensities_valid
        
    
    def overlay_points(self, image, pixel_coords, depths, intensities):
        """叠加点云到图像（使用深度测试的像素级渲染）"""
        result = image.copy()
        
        if len(pixel_coords) == 0:
            return result
        
        alpha = self.alpha_var.get()
        color_mode = self.color_var.get()
        h, w = image.shape[:2]
        
        # 创建深度图和点云图像
        depth_map = np.full((h, w), np.inf, dtype=np.float32)
        pcd_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 转换为整数坐标
        us = pixel_coords[:, 0].astype(np.int32)
        vs = pixel_coords[:, 1].astype(np.int32)
        
        # 边界检查
        valid_mask = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        us_valid = us[valid_mask]
        vs_valid = vs[valid_mask]
        depths_valid = depths[valid_mask]
        
        if len(us_valid) == 0:
            return result
        
        # 生成颜色
        if color_mode == "intensity" and intensities is not None:
            intensities_valid = intensities[valid_mask]
            valid_intensities = intensities_valid[np.isfinite(intensities_valid)]
            
            if len(valid_intensities) > 0 and (valid_intensities.max() - valid_intensities.min()) > 1e-6:
                # 强度着色
                i_min, i_max = valid_intensities.min(), valid_intensities.max()
                norm_i = np.clip((intensities_valid - i_min) / (i_max - i_min), 0.0, 1.0)
                colors_uint8 = (norm_i * 255).astype(np.uint8)
                colors = cv2.applyColorMap(colors_uint8.reshape(-1, 1), cv2.COLORMAP_JET)[:, 0, :]
            else:
                # 回退到深度着色
                color_mode = "depth"
        
        if color_mode == "depth":
            # 深度着色
            d_min, d_max = depths_valid.min(), depths_valid.max()
            if d_max > d_min:
                norm_d = (depths_valid - d_min) / (d_max - d_min)
                colors_uint8 = (norm_d * 255).astype(np.uint8)
                colors = cv2.applyColorMap(colors_uint8.reshape(-1, 1), cv2.COLORMAP_JET)[:, 0, :]
            else:
                colors = np.full((len(us_valid), 3), [0, 255, 0], dtype=np.uint8)
        elif color_mode == "green":
            colors = np.full((len(us_valid), 3), [0, 255, 0], dtype=np.uint8)
        
        # 深度测试渲染（从远到近）
        sort_indices = np.argsort(-depths_valid)  # 降序排序
        us_sorted = us_valid[sort_indices]
        vs_sorted = vs_valid[sort_indices]
        depths_sorted = depths_valid[sort_indices]
        colors_sorted = colors[sort_indices]
        
        # 像素级渲染（支持1x1和2x2像素）
        point_size = 1  # 可以调整为2来获得更大的点
        
        for i in range(len(us_sorted)):
            u, v, depth, color = us_sorted[i], vs_sorted[i], depths_sorted[i], colors_sorted[i]
            
            # 深度测试
            if depth < depth_map[v, u]:
                depth_map[v, u] = depth
                pcd_img[v, u] = color
                
                # 可选：绘制更大的点
                if point_size > 1:
                    for dv in range(-point_size//2, point_size//2 + 1):
                        for du in range(-point_size//2, point_size//2 + 1):
                            nv, nu = v + dv, u + du
                            if 0 <= nv < h and 0 <= nu < w and depth < depth_map[nv, nu]:
                                depth_map[nv, nu] = depth
                                pcd_img[nv, nu] = color
        
        # Alpha混合
        mask = np.any(pcd_img > 0, axis=2)
        result[mask] = cv2.addWeighted(pcd_img[mask], alpha, result[mask], 1 - alpha, 0)
        
        return result


def main():
    root = tk.Tk()
    app = SimplePCDProjection(root)
    root.mainloop()


if __name__ == "__main__":
    main()
