"""
LICENSE.

This project as a whole is licensed under the Apache License, Version 2.0.

THIRD-PARTY LICENSES

Third-party software already included in HoloAgent is governed by the separate
Open Source license terms under which the third-party software has been
distributed.

NOTICE ON LICENSE COMPATIBILITY FOR DISTRIBUTORS

Notably, this project depends on the third-party software FAST-LIVO2 and HOVSG.
Their default licenses restrict commercial use—separate permission from their
original authors is required for commercial integration/redistribution.

The third-party software FAST-LIVO2 dependency (licensed under GPL-2.0-only)
utilizes rpg_vikit-ros2 which contains components under the GPL-3.0. Please be
aware of license compatibility when distributing a combined work.

DISCLAIMER

Users are solely responsible for ensuring compliance with all applicable
license terms when using, modifying, or distributing the project. Project
maintainers accept no liability for any license violations arising from such
use.
"""

#!/usr/bin/env python3
import time
from copy import deepcopy

import hydra
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from hmsg.graph.graph import Graph
from nav2_msgs.action import FollowWaypoints
from omegaconf import DictConfig
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

# pylint: disable=all


class GoalPosePublisher(Node):

    def __init__(self, cfg: DictConfig):
        super().__init__("goal_pose_publisher")

        # Create publisher, message type is PoseStamped, topic name is /goal_pose, queue size is 10
        self.publisher_ = self.create_publisher(PoseStamped, "/object_pose", 10)
        self.waypoint_found_pub = self.create_publisher(String, "waypoint_reached", 10)
        # Subscribe to String topic
        self.subscription = self.create_subscription(
            String, "/chat_loc_pub", self.hmsggetgoal_callback, 10
        )
        self._action_client = ActionClient(self, FollowWaypoints, "/follow_waypoints")
        # Set timer, publish target pose every 1 second
        # timer_period = 1.0  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        self.count = 0
        self.params = cfg
        self.graph = Graph(cfg)
        self.T_switch_axis = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
        )  # g1_navi
        self.T_tomap = np.linalg.inv(self.T_switch_axis)
        self.hmsgcreate()
        self.use_gpt = 0
        # self.hmsggetgoal()

        # Initialize counter

        self.get_logger().info("GoalPosePublisher node started, publishing /object_pose topic...")
        # print(f"This node is running with Python at: {sys.executable}")

    def pubpose(self, x, y, z):
        # Create PoseStamped message
        msg = PoseStamped()

        # Set message header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # Assume target pose is in map coordinate frame

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        # Set target orientation - here use a simple quaternion
        # Make the robot always face the center
        # msg.pose.orientation = self.get_quaternion_from_euler(0, 0, angle + math.pi)

        # Publish message
        self.publisher_.publish(msg)

        # Log
        self.get_logger().info(
            f"Published target pose #{self.count}: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}, z={msg.pose.position.z:.2f}"
        )

        # Increment counter
        self.count += 1

    def hmsgcreate(self):
        # Load graph
        hmsg = self.graph
        hmsg.load_graph(self.params.main.graph_path)
        self.use_gpt = self.params.main.use_gpt
        # Automatically determine room types and names
        # hmsg.generate_room_names(
        #    generate_method="view_embedding",
        #    # digua_demo room_types
        #    default_room_types=[
        #        "Digua Lab",
        #        "Horizon Exhibition Hall",
        #        "Horizon Mini Post Office",
        #        "Long Corridor",
        #        "Corner Corridor",
        #        "Elevator Lobby",
        #        "Elevator",
        #    ]
        # )
        hmsg.generate_room_names(
            generate_method="view_embedding",
            # digua_demo room_types
            default_room_types=[
                "Hallway",
                "Reception area",
                "Exhibition Hall",
                "Pantry",
                "Corner Hallway",
                "Elevator Lobby",
                "Lift",
                "Office",
                "Cafeteria",
            ],
        )
        # Manually set room types and names
        designated_room_names_digua = [
            "none",
            "none",
            "Exhibition Hall",
            "none",
            "Corner Corridor",
            "Corridor",
            "Digua Elevator Lobby Reception Area",
        ]
        designated_room_names_ic7f_demo = [
            "none",
            "none",
            "Office Area",
            "Cafeteria",
            "Elevator Lobby Corridor",
            "Pantry",
            "Office Rest Area",
        ]
        designated_room_names_1014demo = [
            "Corner Corridor",
            "none",
            "Long Corridor",
            "Horizon Exhibition Hall",
            "none",
            "none",
            "Long Corridor",
            "Reception Area",
            "none",
            "Digua Office Area Elevator Lobby",
        ]
        designated_room_names_0918demo = [
            "Reception Area",
        ]

        designated_room_names_1028demo = [
            "none",
            "Meeting Room",
            "Laboratory",
            "none",
            "none",
            "Activity Area",
        ]

        designated_room_names_0918demo = [
            "Reception Area",
        ]

        designated_room_names_1030demo = [
            "Meeting Room",
            "Outdoor",
            "Activity Area",
            "none",
            "none",
            "Operation Area",
            "Meeting Room",
            "Laboratory",
            "Activity Area",
        ]
        designated_room_names_1127demo = [
            "none",
            "Meeting Room",
            "Elevator Lobby",
            "Activity Area",
            "none",
            "Activity Area",
            "none",
        ]
        hmsg.set_room_names(room_names=designated_room_names_1127demo)

    def hmsggetgoal_callback(self, msg):
        hmsg = self.graph
        query_instruction = "From voice search"
        ans = msg.data
        print(ans)
        start_time = time.time()
        floor, room, obj, res_dict = hmsg.query_hierarchy_protected(
            query_instruction, ans, top_k=1, use_gpt=self.use_gpt
        )
        end_time = time.time()
        print("obj: ", res_dict)
        print("score: ", res_dict["object_scores"][0])
        # print(type(res_dict))
        print(f"Run time: {end_time - start_time:.4f} seconds")
        # save log for debug
        # 构建要写入 JSON 的数据
        # query_result = {
        #    "query": query_instruction,
        #    "room_query": res_dict["room_query"],
        #    "object_query": res_dict["object_query"],
        #    "time_seconds": query_time,
        #    "floor_id": floor.floor_id,
        #    "rooms": [{"room_id": r.room_id, "name": r.name} for r in room],
        #    "objects": [{"object_id": o.object_id} for o in obj],
        #    "objects_scores": res_dict["object_scores"]
        # }
        # print(query_result)
        if res_dict["object_query"] != "unknown" and res_dict["object_scores"][0] < 0.15:
            msg = String()
            msg.data = "not_found"
            self.waypoint_found_pub.publish(msg)
            print("not found")
            return
        elif (
            res_dict["room_query"] == "unknown"
            and res_dict["object_query"] == "unknown"
            and res_dict["object_scores"][0] < 0.18
        ):
            return
        else:
            msg = String()
            msg.data = "found"
            self.waypoint_found_pub.publish(msg)
            print("found")

        # visualize the query
        print(floor.floor_id, [(r.room_id, r.name) for r in room], [o.object_id for o in obj])
        # use open3d to visualize room.pcd and color the points where obj.pcd
        # is
        print("len(obj): ", len(obj))
        for i in range(len(obj)):
            obj_pcd = obj[i].pcd.paint_uniform_color([1, 0, 0])  # rgb
            obj_pcd = deepcopy(obj[i].pcd)
            obj_center = obj_pcd.get_center()
            print("obj_center in scenegraph: ", obj_center)
            obj_center_h = np.hstack((obj_center, 1.0))  # 齐次坐标 (4,)
            obj_center_in_map = (self.T_tomap @ obj_center_h)[:3]
            print("obj_center in lidarmap: ", obj_center_in_map)
            self.pubpose(obj_center_in_map[0], obj_center_in_map[1], obj_center_in_map[2])


@hydra.main(
    version_base=None,
    config_path=get_package_share_directory("goal_publisher") + "/config",
    config_name="visualize_query_graph_demo",
)
def main(params: DictConfig, args=None):

    # Initialize ROS2 Python client library
    rclpy.init(args=args)

    # Create node
    goal_pose_publisher = GoalPosePublisher(params)

    try:
        # Run node
        rclpy.spin(goal_pose_publisher)
    except KeyboardInterrupt:
        # Handle Ctrl+C signal
        pass
    finally:
        # Destroy node
        goal_pose_publisher.destroy_node()
        # Shutdown ROS2 Python client library
        rclpy.shutdown()


if __name__ == "__main__":
    main()
