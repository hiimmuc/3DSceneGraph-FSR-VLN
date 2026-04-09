import json
import socket

import cv2
import numpy as np
from tqdm import tqdm


class MapVisual:
    def __init__(self, hmsg, host="127.0.0.1", port=6006):
        self.hmsg = hmsg

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

        self.room_dict = {r.room_id: r for r in self.hmsg.rooms}
        self.floor_dict = {r.floor_id: r for r in self.hmsg.floors}

        self.T_switch_axis = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
        )

        self.T_tomap = np.linalg.inv(self.T_switch_axis)
        self.T_tomap[2, 3] += 1.2

    def send_msg(self, msg_type, data_bytes):
        header = msg_type.to_bytes(1, byteorder="big")
        size = len(data_bytes).to_bytes(8, byteorder="big")
        self.sock.sendall(header + size + data_bytes)

    def colorize(self, points, seed):
        rng = np.random.default_rng(abs(hash(seed)) % (2**32))
        color = rng.random(3)
        colors = np.tile(color, (points.shape[0], 1))
        return np.hstack((points, colors))

    def open3d_to_xyz(self, pcd):
        points = np.asarray(pcd.points)
        ones = np.ones((points.shape[0], 1))
        points_h = np.hstack((points, ones))
        return (self.T_tomap @ points_h.T).T[:, :3].astype(np.float32)

    def publish_highlight_object(self, obj_id):
        obj = self.hmsg.objects[obj_id]

        if len(obj.pcd.points) < 10:
            return

        points = self.open3d_to_xyz(obj.pcd)
        color = np.array([1.0, 0.0, 0.0])
        colors = np.tile(color, (points.shape[0], 1))
        colored = np.hstack((points, colors))

        self.send_msg(4, colored.astype(np.float32).tobytes())

    def compute_room_boundary(self, room):
        verts = np.array(room.vertices, dtype=np.float32)

        if verts.ndim != 2 or len(verts) < 10:
            return None

        # Build a local mask from the contour points
        min_xy = verts.min(axis=0)
        max_xy = verts.max(axis=0)

        scale = 50.0
        size = ((max_xy - min_xy) * scale).astype(int) + 10
        size = np.maximum(size, 1)

        mask = np.zeros((size[1], size[0]), dtype=np.uint8)

        coords = ((verts - min_xy) * scale).astype(int)
        coords = np.clip(coords, 0, np.array([size[0] - 1, size[1] - 1]))

        mask[coords[:, 1], coords[:, 0]] = 255

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)

        boundary = contour.reshape(-1, 2).astype(np.float32)
        boundary = boundary / scale + min_xy

        boundary_3d = np.column_stack(
            [
                boundary[:, 0],
                np.zeros(len(boundary)),
                boundary[:, 1],
            ]
        ).astype(np.float32)

        ones = np.ones((boundary_3d.shape[0], 1))
        boundary_h = np.hstack([boundary_3d, ones])
        boundary_map = (self.T_tomap @ boundary_h.T).T[:, :3]
        boundary_map[:, 2] += 0.05
        boundary_map = np.vstack([boundary_map, boundary_map[0]])

        return boundary_map.astype(np.float32)

    def publish_map(self):
        # 1. Floor
        for i, floor in tqdm(
            enumerate(self.hmsg.floors), desc="Publishing floors", total=len(self.hmsg.floors)
        ):
            points = self.open3d_to_xyz(floor.pcd)
            self.send_msg(0, points.astype(np.float32).tobytes())

        # 2. Rooms + objects merged as colored point cloud
        all_points = []

        pbar_rooms = tqdm(
            enumerate(self.hmsg.rooms), desc="Processing rooms", total=len(self.hmsg.rooms)
        )
        for i, room in pbar_rooms:
            points = self.open3d_to_xyz(room.pcd)
            all_points.append(self.colorize(points, f"room_{i}"))

        pbar_objs = tqdm(
            self.hmsg.objects, desc="Processing objects", total=len(self.hmsg.objects)
        )
        for obj in pbar_objs:
            if len(obj.pcd.points) < 10:
                continue
            points = self.open3d_to_xyz(obj.pcd)
            all_points.append(self.colorize(points, obj.name))

        if all_points:
            self.send_msg(1, np.vstack(all_points).astype(np.float32).tobytes())

        # 3. Object centroids + labels
        pbar_labels = tqdm(
            self.hmsg.objects, desc="Publishing labels", total=len(self.hmsg.objects)
        )
        for obj in pbar_labels:
            center = np.asarray(obj.pcd.points).mean(axis=0)
            center_map = (self.T_tomap @ np.hstack((center, 1.0)))[:3]
            payload = {
                "name": obj.name,
                "x": float(center_map[0]),
                "y": float(center_map[1]),
                "z": float(center_map[2]),
            }
            self.send_msg(2, json.dumps(payload).encode("utf-8"))

        # 4. Scene graph edges
        graph_data = []
        pbar_graph = tqdm(
            self.hmsg.objects, desc="Building scene graph", total=len(self.hmsg.objects)
        )
        for obj in pbar_graph:
            if len(obj.pcd.points) < 10:
                continue
            pbar_graph.set_description_str(f"Building scene graph - Object: {obj.name}")
            center_obj_map = (
                self.T_tomap @ np.hstack((np.asarray(obj.pcd.points).mean(axis=0), 1.0))
            )[:3]
            room = self.room_dict[obj.room_id]
            center_room_map = (
                self.T_tomap @ np.hstack((np.asarray(room.pcd.points).mean(axis=0), 1.0))
            )[:3]
            floor_obj = self.floor_dict[room.floor_id]
            center_floor_map = (
                self.T_tomap @ np.hstack((np.asarray(floor_obj.pcd.points).mean(axis=0), 1.0))
            )[:3]
            graph_data.append(
                {
                    "obj": center_obj_map.tolist(),
                    "room": center_room_map.tolist(),
                    "floor": center_floor_map.tolist(),
                    "name": obj.name,
                }
            )
        pbar_graph.set_description_str("Building scene graph - done")

        self.send_msg(3, json.dumps(graph_data).encode("utf-8"))

        # 5. Room boundaries
        pbar_boundaries = tqdm(
            self.hmsg.rooms, desc="Publishing room boundaries", total=len(self.hmsg.rooms)
        )
        for room in pbar_boundaries:
            boundary = self.compute_room_boundary(room)
            if boundary is None:
                continue
            self.send_msg(5, boundary.tobytes())
