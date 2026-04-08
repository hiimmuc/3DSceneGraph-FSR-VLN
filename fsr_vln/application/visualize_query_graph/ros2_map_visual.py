import json
import random
import socket

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


class TCPPointCloudReceiver(Node):
    def __init__(self):
        super().__init__("tcp_pc_receiver")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_ALL,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        highlight_obj_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_pub = self.create_publisher(PointCloud2, "map_pointcloud", qos)
        self.obj_pub = self.create_publisher(PointCloud2, "obj_pointcloud", qos)
        self.marker_pub = self.create_publisher(Marker, "object_labels", qos)
        self.map2d_marker_pub = self.create_publisher(Marker, "map_2d_marker", qos)
        self.graph_pub = self.create_publisher(Marker, "graph_markers", qos)
        self.highlight_pub = self.create_publisher(
            PointCloud2, "highlight_object", highlight_obj_qos
        )

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", 6006))
        self.server.listen(1)

        print("[TCP] Waiting for connection...")
        self.conn, addr = self.server.accept()
        print(f"[TCP] Connected from {addr}")

        self.marker_id = 0

    def recv_all(self, size):
        data = b""
        while len(data) < size:
            packet = self.conn.recv(size - len(data))
            if not packet:
                return None
            data += packet
        return data

    def receive(self):
        try:
            type_byte = self.recv_all(1)
            if not type_byte:
                return
            msg_type = int.from_bytes(type_byte, "big")

            size_bytes = self.recv_all(8)
            if not size_bytes:
                return
            size = int.from_bytes(size_bytes, "big")

            data = self.recv_all(size)
            if data is None:
                return

            if msg_type == 0:
                xyz = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
                xyz_ds, _ = self.voxel_downsample(xyz, rgb=None, voxel_size=0.08)
                self.map_pub.publish(self.numpy_to_pc2(xyz_ds))

            elif msg_type == 1:
                points = np.frombuffer(data, dtype=np.float32).reshape(-1, 6)
                xyz, rgb = points[:, :3], points[:, 3:]
                xyz_ds, rgb_ds = self.voxel_downsample(xyz, rgb, voxel_size=0.08)
                print(f"[Downsample] {len(xyz)} → {len(xyz_ds)} points")
                self.obj_pub.publish(self.numpy_to_pc2_with_rgb(xyz_ds, rgb_ds))

            elif msg_type == 2:
                self.publish_text_marker(json.loads(data.decode("utf-8")))

            elif msg_type == 3:
                self.publish_graph(json.loads(data.decode("utf-8")))

            elif msg_type == 4:
                points = np.frombuffer(data, dtype=np.float32).reshape(-1, 6)
                self.highlight_pub.publish(
                    self.numpy_to_pc2_with_rgb(points[:, :3], points[:, 3:])
                )

            elif msg_type == 5:
                points = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
                self.publish_room_polygon(points)

        except Exception as e:
            print("[ERROR]", e)

    def publish_room_polygon(self, points):
        if len(points) < 3:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "room_fill"
        marker.id = self.marker_id
        self.marker_id += 1

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05

        marker.color.r = random.random()
        marker.color.g = random.random()
        marker.color.b = random.random()
        marker.color.a = 1.0

        for p in points:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = 0.0
            marker.points.append(pt)

        marker.points.append(marker.points[0])
        self.map2d_marker_pub.publish(marker)

    def publish_graph(self, graph):
        stamp = self.get_clock().now().to_msg()

        floor_marker = Marker()
        floor_marker.header.frame_id = "map"
        floor_marker.header.stamp = stamp
        floor_marker.ns = "floor_nodes"
        floor_marker.id = 0
        floor_marker.type = Marker.SPHERE_LIST
        floor_marker.action = Marker.ADD
        floor_marker.scale.x = 0.5
        floor_marker.scale.y = 0.5
        floor_marker.scale.z = 0.5
        floor_marker.color.r = 1.0
        floor_marker.color.g = 0.0
        floor_marker.color.b = 0.0
        floor_marker.color.a = 1.0

        room_marker = Marker()
        room_marker.header.frame_id = "map"
        room_marker.header.stamp = stamp
        room_marker.ns = "room_nodes"
        room_marker.id = 1
        room_marker.type = Marker.SPHERE_LIST
        room_marker.action = Marker.ADD
        room_marker.scale.x = 0.3
        room_marker.scale.y = 0.3
        room_marker.scale.z = 0.3
        room_marker.color.r = 0.0
        room_marker.color.g = 0.0
        room_marker.color.b = 1.0
        room_marker.color.a = 1.0

        obj_marker = Marker()
        obj_marker.header.frame_id = "map"
        obj_marker.header.stamp = stamp
        obj_marker.ns = "object_nodes"
        obj_marker.id = 2
        obj_marker.type = Marker.SPHERE_LIST
        obj_marker.action = Marker.ADD
        obj_marker.scale.x = 0.15
        obj_marker.scale.y = 0.15
        obj_marker.scale.z = 0.15
        obj_marker.color.r = 0.0
        obj_marker.color.g = 1.0
        obj_marker.color.b = 0.0
        obj_marker.color.a = 1.0

        edge_marker = Marker()
        edge_marker.header.frame_id = "map"
        edge_marker.header.stamp = stamp
        edge_marker.ns = "graph_edges"
        edge_marker.id = 3
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.01
        edge_marker.color.r = 1.0
        edge_marker.color.g = 1.0
        edge_marker.color.b = 1.0
        edge_marker.color.a = 0.3

        for item in graph:
            p_obj = Point(
                x=float(item["obj"][0]), y=float(item["obj"][1]), z=float(item["obj"][2])
            )
            p_room = Point(
                x=float(item["room"][0]), y=float(item["room"][1]), z=float(item["room"][2])
            )

            if "floor" in item:
                p_floor = Point(
                    x=float(item["floor"][0]), y=float(item["floor"][1]), z=float(item["floor"][2])
                )
                floor_marker.points.append(p_floor)
                edge_marker.points.append(p_floor)
                edge_marker.points.append(p_room)

            room_marker.points.append(p_room)
            obj_marker.points.append(p_obj)
            edge_marker.points.append(p_obj)
            edge_marker.points.append(p_room)

        self.graph_pub.publish(floor_marker)
        self.graph_pub.publish(room_marker)
        self.graph_pub.publish(obj_marker)
        self.graph_pub.publish(edge_marker)

    def numpy_to_pc2(self, points):
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()
        return point_cloud2.create_cloud_xyz32(header, points)

    def numpy_to_pc2_with_rgb(self, xyz, rgb):
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()

        rgb_uint8 = (rgb * 255).astype(np.uint8)
        rgb_packed = (
            (rgb_uint8[:, 0].astype(np.uint32) << 16)
            | (rgb_uint8[:, 1].astype(np.uint32) << 8)
            | rgb_uint8[:, 2].astype(np.uint32)
        )

        points = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        points[:, :3] = xyz
        points[:, 3] = rgb_packed.view(np.float32)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return point_cloud2.create_cloud(header, fields, points)

    def voxel_downsample(self, xyz, rgb=None, voxel_size=0.08):
        coords = np.floor(xyz / voxel_size)
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

        xyz_ds = np.zeros((len(unique_coords), 3), dtype=np.float32)
        rgb_ds = np.zeros((len(unique_coords), 3), dtype=np.float32)
        counts = np.zeros(len(unique_coords))

        for i in range(len(xyz)):
            idx = inverse[i]
            xyz_ds[idx] += xyz[i]
            if rgb is not None:
                rgb_ds[idx] += rgb[i]
            counts[idx] += 1

        xyz_ds /= counts[:, None]
        if rgb is not None:
            rgb_ds /= counts[:, None]

        return xyz_ds, rgb_ds

    def publish_text_marker(self, data):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.lifetime = Duration(sec=0)
        marker.ns = "labels"
        marker.id = hash((data["name"], data["x"], data["y"], data["z"])) % 100000
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.text = data["name"]

        marker.pose.position.x = data["x"]
        marker.pose.position.y = data["y"]
        marker.pose.position.z = data["z"] + 0.3

        marker.scale.z = 0.07

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    def receive_loop(self):
        while rclpy.ok():
            self.receive()


def main():
    rclpy.init()
    node = TCPPointCloudReceiver()

    try:
        node.receive_loop()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down cleanly...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
