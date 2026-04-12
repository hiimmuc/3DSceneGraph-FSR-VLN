#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机位姿插值工具 - GUI版本
功能：根据imu-pose和Til,Tcl外参，生成相机位姿文件
输入：img_pos_opt.txt + cam_in_ex_opt.txt
输出：cam_0_pos.txt + cam_1_pos.txt + cam_2_pos.txt
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import platform
import numpy as np
import threading
import subprocess
from scipy.spatial.transform import Rotation as R

# image: 1600x1296
# stereo_image: 1920x1080

class CameraPoseInterpolationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("相机位姿插值工具")
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
        
        self.output_dir = "/mnt/disk1/mapvln/pocket2/"
        
        # 数据
        self.trigger_times = []  # [(id, timestamp), ...]
        self.img_poses = []  # [(timestamp, x, y, z, qw, qx, qy, qz), ...]
        self.cameras_extrinsics = {}  # {cam_id: Tcl_matrix}
        
        # os.makedirs(self.output_dir, exist_ok=True)
        self.create_widgets()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 标题（使用可伸缩的字体）
        ttk.Label(self.root, text="相机位姿插值工具", font=self.title_font).pack(pady=10)
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文件选择
        file_frame = ttk.LabelFrame(main_frame, text="输入文件选择", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        # trigger_times.txt
        self.create_file_row(file_frame, "触发时间戳:", "trigger_entry", self.select_trigger)
        
        # img_pos.txt
        self.create_file_row(file_frame, "IMU位姿文件:", "imgpos_entry", self.select_imgpos)
        
        # cam_in_ex.txt
        self.create_file_row(file_frame, "相机外参文件:", "calib_entry", self.select_calib)
        
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
        
        self.info_text = tk.Text(info_frame, height=5, width=70, wrap=tk.WORD, state='disabled')
        self.info_text.pack(fill=tk.X)
        
        # 插值参数
        param_frame = ttk.LabelFrame(main_frame, text="插值参数", padding="10")
        param_frame.pack(fill=tk.X, pady=10)
        
        # 显示参数信息
        param_info = ttk.Label(param_frame, 
                              text="插值方法：平移-线性插值 | 旋转-四元数SLERP\n"
                                   "假设：IMU和LiDAR坐标系重合 (T_i_l = I)",
                              foreground="blue")
        param_info.pack(pady=5)
        
        # 按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.process_btn = ttk.Button(btn_frame, text="开始插值", command=self.start_process)
        self.process_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(btn_frame, text="打开结果", command=self.open_output).pack(side=tk.LEFT, padx=5)
        
        # 进度
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.status_var = tk.StringVar(value="就绪 - 请选择输入文件")
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
    
    def select_trigger(self):
        """选择trigger_times.txt"""
        path = filedialog.askopenfilename(
            title="选择触发时间戳文件",
            filetypes=[("文本文件", "trigger_times.txt *.txt"), ("所有文件", "*.*")]
        )
        if path:
            self.trigger_entry.delete(0, tk.END)
            self.trigger_entry.insert(0, path)
            self.load_trigger_times(path)
    
    def select_imgpos(self):
        """选择img_pos.txt"""
        path = filedialog.askopenfilename(
            title="选择IMU位姿文件",
            filetypes=[("文本文件", "img_pos.txt *.txt"), ("所有文件", "*.*")]
        )
        # self.output_dir = os.path.dirname(path)
        if path:
            self.imgpos_entry.delete(0, tk.END)
            self.imgpos_entry.insert(0, path)
            self.load_img_poses(path)
    
    def select_calib(self):
        """选择cam_in_ex.txt"""
        path = filedialog.askopenfilename(
            title="选择相机外参文件",
            filetypes=[("文本文件", "cam_in_ex.txt *.txt"), ("所有文件", "*.*")]
        )
        if path:
            self.calib_entry.delete(0, tk.END)
            self.calib_entry.insert(0, path)
            self.load_camera_extrinsics(path)
    
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
    
    def load_trigger_times(self, path):
        """加载触发时间戳"""
        self.log(f"加载触发时间戳: {os.path.basename(path)}")
        try:
            self.trigger_times = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        trigger_id = int(parts[0])
                        timestamp = float(parts[1])
                        self.trigger_times.append((trigger_id, timestamp))
            
            self.log(f"  ✓ 加载 {len(self.trigger_times)} 个触发时间戳")
            if self.trigger_times:
                t_min = self.trigger_times[0][1]
                t_max = self.trigger_times[-1][1]
                self.log(f"  时间范围: {t_min:.6f} - {t_max:.6f}")
            
            self.update_file_info()
            
        except Exception as e:
            self.log(f"  ✗ 失败: {e}")
            messagebox.showerror("错误", f"加载触发时间戳失败:\n{e}")
    
    def load_img_poses(self, path):
        """加载IMU位姿"""
        self.log(f"加载IMU位姿: {os.path.basename(path)}")
        try:
            self.img_poses = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 9:
                        timestamp = float(parts[1])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        qw, qx, qy, qz = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                        self.img_poses.append((timestamp, x, y, z, qw, qx, qy, qz))
            
            self.log(f"  ✓ 加载 {len(self.img_poses)} 个IMU位姿")
            if self.img_poses:
                t_min = self.img_poses[0][0]
                t_max = self.img_poses[-1][0]
                self.log(f"  时间范围: {t_min:.6f} - {t_max:.6f}")
            
            self.update_file_info()
            
        except Exception as e:
            self.log(f"  ✗ 失败: {e}")
            messagebox.showerror("错误", f"加载IMU位姿失败:\n{e}")
    
    def load_camera_extrinsics(self, path):
        """加载相机外参"""
        self.log(f"加载相机外参: {os.path.basename(path)}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析Tcl矩阵
            tcl_pattern = r'Tcl_(\d+):\s*\[([^\]]+)\]'
            tcl_matches = re.findall(tcl_pattern, content, re.MULTILINE | re.DOTALL)
            
            self.cameras_extrinsics = {}
            for cam_idx, values_str in tcl_matches:
                values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)
                
                if len(values) >= 16:
                    Tcl = np.array([float(v) for v in values[:16]]).reshape(4, 4)
                    cam_id = f'cam_{cam_idx}'
                    self.cameras_extrinsics[cam_id] = Tcl
                    self.log(f"  ✓ 加载 {cam_id} 外参")

            # 解析Til矩阵
            til_pattern = r'Til:\s*\[([^\]]+)\]'
            til_match = re.search(til_pattern, content, re.MULTILINE | re.DOTALL)
            if til_match:
                til_values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', til_match.group(1))
                if len(til_values) >= 16:
                    self.Til = np.array([float(v) for v in til_values[:16]]).reshape(4, 4)
                    self.log("  ✓ 加载 Til 矩阵")
            else:
                self.Til = None
                self.log("  ⚠ 未找到 Til 矩阵")
            print("self.Til:", self.Til)

            # 解析相机内参
            cam_pattern = r'(cam_\d+):\s*((?:\s+\w+:.*\n)+)'
            cam_matches = re.findall(cam_pattern, content)
            self.cameras_intrinsics = {}
            for cam_id, cam_str in cam_matches:
                intrinsics = {}
                lines = cam_str.strip().splitlines()
                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        try:
                            intrinsics[key.strip()] = float(value.strip())
                        except ValueError:
                            intrinsics[key.strip()] = value.strip()
                self.cameras_intrinsics[cam_id] = intrinsics
                self.log(f"  ✓ 加载 {cam_id} 内参")
                print(f"{self.cameras_intrinsics[cam_id]}:", self.cameras_intrinsics[cam_id])

            
            if not self.cameras_extrinsics:
                raise Exception("未找到有效的相机外参")
            
            self.update_file_info()
            
        except Exception as e:
            self.log(f"  ✗ 失败: {e}")
            messagebox.showerror("错误", f"加载外参失败:\n{e}")
    
    def update_file_info(self):
        """更新文件信息显示"""
        info = []
        
        if self.trigger_times:
            info.append(f"触发时间戳: {len(self.trigger_times)} 个")
        
        if self.img_poses:
            info.append(f"IMU位姿: {len(self.img_poses)} 个")
        
        if self.cameras_extrinsics:
            cams = ', '.join(sorted(self.cameras_extrinsics.keys()))
            info.append(f"相机外参: {cams}")
        
        if info:
            self.update_info('\n'.join(info))
    
    def start_process(self):
        """开始处理"""
        # 检查输入
        if not self.trigger_entry.get() or not self.trigger_times:
            messagebox.showerror("错误", "请先选择触发时间戳文件")
            return
        
        if not self.imgpos_entry.get() or not self.img_poses:
            messagebox.showerror("错误", "请先选择IMU位姿文件")
            return
        
        if not self.calib_entry.get() or not self.cameras_extrinsics:
            messagebox.showerror("错误", "请先选择相机外参文件")
            return
        
        self.process_btn.config(state='disabled')
        self.progress_bar.start()
        
        thread = threading.Thread(target=self.process, daemon=True)
        thread.start()
    
    def process(self):
        """处理流程"""
        try:
            self.log("="*60)
            self.log("开始插值处理...")
            self.status_var.set("插值中...")
            
            os.makedirs(self.output_dir, exist_ok=True)
            # T_updatedW_originW = np.loadtxt("/mnt/disk1/mapvln/pocket2/optimized_transformation_taget2source.txt")
            
            # 遍历每个相机
            for cam_id, Tcl in self.cameras_extrinsics.items():
                # output_file = os.path.join(self.output_dir, f"{cam_id}_poses.txt")
                output_file = os.path.join(self.output_dir, f"{cam_id}_pos.txt")
                
                with open(output_file, 'w') as f:
                    # for trigger_id, timestamp in self.trigger_times:
                    for trigger_id, timestamp in self.trigger_times:
                        # 插值IMU位姿， imu2world
                        # T_w_imu = self.interpolate_pose(timestamp)

                        timestamp, T_w_imu = self.no_interpolate_pose(trigger_id)
                        
                        # 计算相机位姿, cam2world
                        T_originw_cam = self.compute_camera_pose(T_w_imu, Tcl)

                        # 转换到目标坐标系
                        # T_w_cam = T_updatedW_originW @ T_originw_cam

                        T_w_cam = T_originw_cam
                        
                        # 提取平移和旋转
                        translation = T_w_cam[:3, 3]
                        rotation_matrix = T_w_cam[:3, :3]
                        
                        # 转换为四元数
                        rot = R.from_matrix(rotation_matrix)
                        quat = rot.as_quat()  # [qx, qy, qz, qw]
                        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                        
                        # 写入
                        f.write(f"{trigger_id} {timestamp:.10f} ")
                        # f.write(f"{trigger_id:06d} ")
                        f.write(f"{translation[0]:.10f} {translation[1]:.10f} {translation[2]:.10f} ")
                        f.write(f"{qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f}\n")
                
                self.log(f"  ✓ 导出 {cam_id}: {os.path.basename(output_file)}")
                self.log(f"    共 {len(self.trigger_times)} 帧")
            
            self.log("="*60)
            self.log("✓ 插值完成！")
            self.status_var.set("完成")
            
            result_msg = f"相机位姿插值完成！\n\n输出目录:\n{self.output_dir}\n\n生成文件:\n"
            for cam_id in sorted(self.cameras_extrinsics.keys()):
                result_msg += f"- {cam_id}_pos.txt\n"
            
            messagebox.showinfo("成功", result_msg)
            
        except Exception as e:
            self.log(f"✗ 失败: {e}")
            self.status_var.set("失败")
            messagebox.showerror("错误", f"插值失败:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            self.progress_bar.stop()
            self.process_btn.config(state='normal')
    
    def interpolate_pose(self, target_timestamp):
        """插值IMU位姿"""
        timestamps = [pose[0] for pose in self.img_poses]
        
        # 边界情况
        if target_timestamp <= timestamps[0]:
            _, x, y, z, qw, qx, qy, qz = self.img_poses[0]
        elif target_timestamp >= timestamps[-1]:
            _, x, y, z, qw, qx, qy, qz = self.img_poses[-1]
        else:
            # 找到插值区间
            idx = 0
            for i in range(len(timestamps) - 1):
                if timestamps[i] <= target_timestamp <= timestamps[i + 1]:
                    idx = i
                    break
            
            t1, x1, y1, z1, qw1, qx1, qy1, qz1 = self.img_poses[idx]
            t2, x2, y2, z2, qw2, qx2, qy2, qz2 = self.img_poses[idx + 1]
            
            # 插值系数
            alpha = (target_timestamp - t1) / (t2 - t1)
            
            # 平移线性插值
            x = x1 + alpha * (x2 - x1)
            y = y1 + alpha * (y2 - y1)
            z = z1 + alpha * (z2 - z1)
            
            # 四元数SLERP
            q1 = np.array([qw1, qx1, qy1, qz1])
            q2 = np.array([qw2, qx2, qy2, qz2])
            q_interp = self.quaternion_slerp(q1, q2, alpha)
            qw, qx, qy, qz = q_interp
        
        # 构建变换矩阵
        rot = R.from_quat([qx, qy, qz, qw])
        T_w_imu = np.eye(4)
        T_w_imu[:3, :3] = rot.as_matrix()
        T_w_imu[:3, 3] = [x, y, z]
        
        return T_w_imu
    
    def no_interpolate_pose(self, idx):
        img_timestamp, x, y, z, qw, qx, qy, qz = self.img_poses[idx]
        # 构建变换矩阵
        rot = R.from_quat([qx, qy, qz, qw])
        T_w_imu = np.eye(4)
        T_w_imu[:3, :3] = rot.as_matrix()
        T_w_imu[:3, 3] = [x, y, z]

        return img_timestamp, T_w_imu
    
    def quaternion_slerp(self, q1, q2, t):
        """四元数球面线性插值"""
        dot = np.dot(q1, q2)
        
        # 选择最短路径
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # 如果非常接近，使用线性插值
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # SLERP
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        theta = theta_0 * t
        
        q2_orth = q2 - q1 * dot
        q2_orth = q2_orth / np.linalg.norm(q2_orth)
        
        return q1 * np.cos(theta) + q2_orth * np.sin(theta)
    
    def compute_camera_pose(self, T_w_imu, Tcl):
        """计算相机位姿 camera2world""" 
        # 假设IMU和LiDAR重合
        
        T_i_l = np.array([
            [1.0, 0.0, 0.0, -0.011],
            [0.0, 1.0, 0.0, -0.02329],
            [0.0, 0.0, 1.0, 0.04412],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        T_i_l = self.Til.astype(np.float64)
        # T_w_cam = (T_w_imu * T_i_l) * inv(Tcl)
        Tcl_inv = np.linalg.inv(Tcl)
        T_w_cam = (T_w_imu @ T_i_l) @ Tcl_inv
        
        return T_w_cam


def main():
    root = tk.Tk()
    app = CameraPoseInterpolationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
