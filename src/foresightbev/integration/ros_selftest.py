"""Verify real ROS 2 message round trips and conventions. Run inside the pixi environment."""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import String

from foresightbev.contracts import CellState
from foresightbev.integration import ros_messages as M
from foresightbev.integration.ros_common import RELIABLE


def exchange(node, kind, topic: str, message):
    received = []
    subscription = node.create_subscription(kind, topic, received.append, RELIABLE)
    publisher = node.create_publisher(kind, topic, RELIABLE)
    deadline = time.monotonic() + 10
    while not received and time.monotonic() < deadline:
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(subscription)
    node.destroy_publisher(publisher)
    if not received:
        raise TimeoutError(f"nothing received on {topic}")
    return received[0]


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("foresight_selftest")
    checks: dict[str, bool] = {}
    try:
        labels = np.arange(30, dtype=np.uint8).reshape(5, 6) % 8
        image = Image(height=5, width=6, step=6, encoding="mono8", data=labels.tobytes())
        back = exchange(node, Image, "/selftest/image", image)
        checks["image_mono8_is_bit_exact"] = (np.frombuffer(bytes(back.data), np.uint8).reshape(back.height, back.width) == labels).all().item()

        odometry = Odometry()
        sec, nanosec = M.to_stamp(12.345678901)
        odometry.header.stamp.sec, odometry.header.stamp.nanosec = sec, nanosec
        o = odometry.pose.pose.orientation
        o.x, o.y, o.z, o.w = M.yaw_to_quaternion(-2.5)
        back = exchange(node, Odometry, "/selftest/odometry", odometry)
        b = back.pose.pose.orientation
        checks["odometry_yaw_survives"] = math.isclose(M.quaternion_to_yaw(b.x, b.y, b.z, b.w), -2.5, abs_tol=1e-9)
        checks["stamp_keeps_nanoseconds"] = M.stamp_key(back.header.stamp.sec, back.header.stamp.nanosec) == M.stamp_key(sec, nanosec)

        state = np.full((4, 3), int(CellState.UNKNOWN), np.uint8)
        state[2, 1] = CellState.OCCUPIED
        state[3, 0] = CellState.FREE
        grid = OccupancyGrid()
        grid.info.width, grid.info.height, grid.info.resolution = 4, 3, 0.25
        grid.info.origin.position.x, grid.info.origin.position.y = -2.0, -1.0
        grid.data = M.occupancy_grid_data(state).tolist()
        back = exchange(node, OccupancyGrid, "/selftest/grid", grid)
        data = np.array(back.data, np.int8).reshape(back.info.height, back.info.width)
        checks["occupancy_row_major_x_along_width"] = int(data[1, 2]) == 100 and int(data[0, 3]) == 0 and int(data[0, 0]) == -1

        path = Path()
        item = PoseStamped()
        item.pose.position.x = 3.5
        item.pose.position.y = -1.25
        path.poses.append(item)
        back = exchange(node, Path, "/selftest/path", path)
        checks["path_positions_survive"] = back.poses[0].pose.position.x == 3.5 and back.poses[0].pose.position.y == -1.25

        back = exchange(node, String, "/selftest/string", String(data=json.dumps({"a": [1, 2]})))
        checks["json_string_survives"] = json.loads(back.data) == {"a": [1, 2]}
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    print(json.dumps(dict(ros_distro=os.environ.get("ROS_DISTRO"), checks=checks), indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
