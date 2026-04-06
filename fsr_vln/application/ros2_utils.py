"""ROS2 utilities for HoloAgent Streamlit UI.

Provides a background-thread camera subscriber for /camera/rgb so the Streamlit
app can display live robot camera frames without blocking the UI.

All ROS2 imports are guarded; the module loads cleanly without a ROS2 installation.
"""

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_camera_lock = threading.Lock()
_latest_frame: Optional[np.ndarray] = None
_subscriber_thread: Optional[threading.Thread] = None
_ros_running = False


def is_ros2_available() -> bool:
    """Return True if rclpy is importable."""
    try:
        import rclpy  # noqa: F401
        return True
    except ImportError:
        return False


def get_latest_frame() -> Optional[np.ndarray]:
    """Return the most recent camera frame as a BGR numpy array, or None."""
    with _camera_lock:
        return _latest_frame.copy() if _latest_frame is not None else None


def start_camera_subscriber(topic: str = "/camera/rgb") -> bool:
    """Start background thread that subscribes to a ROS2 image topic.

    Args:
        topic: ROS2 topic name (default ``/camera/rgb``).

    Returns:
        True if the subscriber was started successfully, False otherwise.
    """
    global _subscriber_thread, _ros_running

    if not is_ros2_available():
        logger.warning("ROS2 not available — camera stream disabled.")
        return False

    if _subscriber_thread is not None and _subscriber_thread.is_alive():
        logger.debug("Camera subscriber already running.")
        return True

    _ros_running = True
    _subscriber_thread = threading.Thread(
        target=_ros2_spin_loop,
        args=(topic,),
        daemon=True,
        name="ros2_camera_subscriber",
    )
    _subscriber_thread.start()
    logger.info("Camera subscriber started on topic '%s'.", topic)
    return True


def stop_camera_subscriber() -> None:
    """Signal the camera subscriber thread to stop."""
    global _ros_running
    _ros_running = False


def _ros2_spin_loop(topic: str) -> None:
    """Internal: spin the ROS2 node in a background thread."""
    global _ros_running, _latest_frame

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        try:
            from cv_bridge import CvBridge
            _bridge = CvBridge()
        except ImportError:
            logger.warning("cv_bridge not found — falling back to manual image conversion.")
            _bridge = None

        rclpy.init()

        class _CameraNode(Node):
            def __init__(self):
                super().__init__("holoagent_camera_viewer")
                self.sub = self.create_subscription(Image, topic, self._cb, 10)

            def _cb(self, msg: Image):
                global _latest_frame
                try:
                    if _bridge is not None:
                        frame = _bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    else:
                        frame = _manual_convert(msg)
                    with _camera_lock:
                        _latest_frame = frame
                except Exception as e:
                    logger.debug("Camera frame conversion error: %s", e)

        node = _CameraNode()

        while _ros_running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        logger.error("ROS2 camera subscriber error: %s", e)


def _manual_convert(msg) -> np.ndarray:
    """Fallback image conversion without cv_bridge."""
    import numpy as np

    data = np.frombuffer(msg.data, dtype=np.uint8)
    h, w = msg.height, msg.width

    if msg.encoding in ("rgb8", "bgr8"):
        frame = data.reshape(h, w, 3)
        if msg.encoding == "rgb8":
            frame = frame[:, :, ::-1]  # RGB → BGR
    elif msg.encoding == "rgba8":
        frame = data.reshape(h, w, 4)[:, :, :3][:, :, ::-1]
    elif msg.encoding in ("mono8", "8UC1"):
        frame = np.stack([data.reshape(h, w)] * 3, axis=-1)
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    return frame
