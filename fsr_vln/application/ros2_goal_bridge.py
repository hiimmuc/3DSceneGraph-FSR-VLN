import json
import socket

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class GoalBridgeNode(Node):
    def __init__(self):
        super().__init__("hmsg_goal_bridge")
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_position", 10)
        self.marker_pub = self.create_publisher(Marker, "/goal_marker", 10)

        # Setup UDP socket listener
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 5005))
        self.sock.setblocking(False)  # Non-blocking so ROS can spin

        # Check for new data at 10Hz
        self.timer = self.create_timer(0.1, self.check_udp_data)
        self.get_logger().info("UDP to ROS 2 Bridge is running. Listening on port 5005...")

    def check_udp_data(self):
        try:
            data, addr = self.sock.recvfrom(1024)
            goal_info = json.loads(data.decode("utf-8"))
            self.publish_ros_goal(goal_info)
        except BlockingIOError:
            pass  # No data received yet
        except Exception as e:
            self.get_logger().error(f"Error receiving data: {e}")

    def publish_ros_goal(self, goal_info):
        # 1. PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = goal_info["x"]
        pose_msg.pose.position.y = goal_info["y"]
        pose_msg.pose.position.z = goal_info["z"]
        pose_msg.pose.orientation.w = 1.0

        # 2. Marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "hmsg_query_goals"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        marker.lifetime.sec = 0

        self.goal_pub.publish(pose_msg)
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published goal for: {goal_info['name']}")


def main(args=None):
    rclpy.init(args=args)
    node = GoalBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
