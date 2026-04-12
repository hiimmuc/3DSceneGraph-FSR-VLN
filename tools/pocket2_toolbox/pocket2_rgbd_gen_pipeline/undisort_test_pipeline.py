import os
import cv2
import numpy as np
from tqdm import tqdm  # 可选，用于显示进度

# ===== 参数 =====MT20251211-195707
# src_dir = "/mnt/disk1/mapvln/pocket2/MT20251211-195707/image/cam1"
# dst_dir = "/mnt/disk1/mapvln/pocket2/MT20251211-195707/image/rectify_cam1"
# os.makedirs(dst_dir, exist_ok=True)
# # ===== 高阶鱼眼 + 原相机内参 digua_stereo_pocket2-cam1=====
# fx_src = 734.504       # A11
# fy_src = 734.719       # A22
# cx_src = 787.674       # u0
# cy_src = 628.856       # v0
# skew = -0.485388  # A12
# # 高阶鱼眼畸变系数
# k2 = -0.0128286
# k3 = 0.0314917
# k4 = -0.0903
# k5 = 0.0923087
# k6 = -0.0511442
# k7 = 0.0102951
# # ===== 高阶鱼眼 + 原相机内参 量产_pocket2-cam1 =====

def load_camera_params(cam_name, calib_file):
    params = {}
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    cam_section = False
    for line in lines:
        line_strip = line.strip()
        # 进入目标相机参数块
        if line_strip.startswith(f'{cam_name}:'):
            cam_section = True
            continue
        # 离开参数块：遇到下一个 cam_x:、Til:、空行、或以冒号结尾的行
        if cam_section:
            if line_strip == '' or line_strip.endswith(':') or line_strip.startswith('Til:') or (line_strip.startswith('cam_') and line_strip.endswith(':')):
                break
            if ':' not in line:
                continue
            key, value = line_strip.split(':', 1)
            value = value.strip()
            # 跳过非标量（如 Til: [..] 或带逗号的列表）
            if value.startswith('[') or ',' in value:
                continue
            try:
                params[key.strip()] = float(value)
            except ValueError:
                continue
    return params

# 自动加载参数
cam_name = 'cam_1'  # 可改为 cam_0 或 cam_1
# # ===== 参数 =====MMT20260108-174617
img_dir = "/mnt/disk1/mapvln/pocket2/MT20260309-104919/image"
src_dir = os.path.join(img_dir, cam_name)
dst_dir = os.path.join(img_dir, "rectify_" + cam_name)
calib_file = os.path.join(img_dir, "cam_in_ex_opt.txt")
os.makedirs(dst_dir, exist_ok=True)
params = load_camera_params(cam_name, calib_file)
fx_src = params['A11']
fy_src = params['A22']
cx_src = params['u0']
cy_src = params['v0']
skew = params['A12']
k2 = params['k2']
k3 = params['k3']
k4 = params['k4']
k5 = params['k5']
k6 = params['k6']
k7 = params['k7']
p1 = params['p1']
p2 = params['p2']

# 输出尺寸和缩放比例
scale = 0.4
dst_width = 640
dst_height = 480

# ===== 构建 remap =====
def compute_map(src_width, src_height, dst_width, dst_height, scale):
    fx = fx_src * scale
    fy = fy_src * scale
    cx = cx_src * scale
    cy = cy_src * scale

    mapx = np.zeros((dst_height, dst_width), dtype=np.float32)
    mapy = np.zeros((dst_height, dst_width), dtype=np.float32)

    for v in range(dst_height):
        for u in range(dst_width):
            x = (u - cx) / fx
            y = (v - cy) / fy
            r = np.sqrt(x*x + y*y)
            if r < 1e-8:
                mapx[v,u] = cx_src
                mapy[v,u] = cy_src
                continue
            theta = np.arctan(r)
            theta2 = theta*theta
            theta3 = theta2*theta
            theta4 = theta3*theta
            theta5 = theta4*theta
            theta6 = theta5*theta
            theta7 = theta6*theta
            theta_d = theta + k2*theta2 + k3*theta3 + k4*theta4 + k5*theta5 + k6*theta6 + k7*theta7
            scale_theta = theta_d / r
            xd = x * scale_theta
            yd = y * scale_theta
            mapx[v,u] = fx_src * xd + skew * yd + cx_src
            mapy[v,u] = fy_src * yd + cy_src

    return mapx, mapy

# ===== 获取源图像尺寸（任意一张） =====
sample_img = cv2.imread(os.path.join(src_dir, os.listdir(src_dir)[0]))
src_h, src_w = sample_img.shape[:2]

mapx, mapy = compute_map(src_w, src_h, dst_width, dst_height, scale)
K_dist = np.array([[fx_src * scale, 0, cx_src * scale],
              [0, fy_src * scale, cy_src * scale],
              [0, 0, 1]])

# ===== 遍历处理所有图像 =====
for id, fname in tqdm(enumerate(sorted(os.listdir(src_dir)))):
    if not fname.lower().endswith((".jpg",".jpeg",".png")):
        continue

    img_path = os.path.join(src_dir, fname)
    frame_id = int(fname.split(".")[0].split("_")[-1])
    img = cv2.imread(img_path)
    if img is None:
        print(f"跳过 {fname}，加载失败")
        continue

    undistorted = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)


    # save_path = os.path.join(dst_dir, fname)
    save_path = os.path.join(dst_dir, f"{frame_id:06d}.jpg")
    cv2.imwrite(save_path, undistorted)
    # print(f"已处理 {fname}，保存至 {save_path}")
print("K_dist: ", K_dist)
# 保存 K_dist 到 rectify_cam_id_param.txt
param_save_path = os.path.join(img_dir, f"rectify_{cam_name}_param.txt")
np.savetxt(param_save_path, K_dist, fmt='%.6f')
print(f"K_dist 已保存到 {param_save_path}，可直接用 np.loadtxt 读取")