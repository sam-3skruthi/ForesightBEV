"""Shared ROS 2 publishing. Requires rclpy, so it runs only in the pixi environment."""

from __future__ import annotations

import json
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from foresightbev.contracts import CalibrationBundle, FrameBundle
from foresightbev.integration import ros_messages as M
from foresightbev.risk.trajectories import RouteContext

RELIABLE = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)


def announce_done(node, publisher, seconds: float = 1.5) -> None:
    """Repeat the end signal while spinning, so it survives this process exiting."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        publisher.publish(String(data="done"))
        rclpy.spin_once(node, timeout_sec=0.1)


class FramePublisher:
    """Publishes one synchronised frame on the topics the pipeline node subscribes to."""

    def __init__(self, node, calibration: CalibrationBundle):
        self.images = {c.name: node.create_publisher(Image, f"/foresight/cameras/{c.name}/labels", RELIABLE) for c in calibration.cameras}
        self.ego = node.create_publisher(Odometry, "/foresight/ego", RELIABLE)
        self.route = node.create_publisher(String, "/foresight/route", RELIABLE)
        self.ready = False
        node.create_subscription(String, "/foresight/pipeline_ready", self._on_ready, RELIABLE)

    def _on_ready(self, message: String) -> None:
        self.ready = True

    def peers_ready(self) -> bool:
        """True once the pipeline confirms its readers match all of our writers.

        A writer can see a reader before that reader sees the writer. With volatile
        QoS anything sent in that gap is lost, so a writer-side count is not enough.
        """
        return self.ready and all(p.get_subscription_count() > 0 for p in self.images.values()) and self.route.get_subscription_count() > 0 and self.ego.get_subscription_count() > 0

    def publish(self, bundle: FrameBundle, route: RouteContext, skip_camera: str | None = None) -> None:
        sec, nanosec = M.to_stamp(bundle.timestamp)
        stamp = rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()
        self.route.publish(String(data=json.dumps(dict(frame=bundle.frame_index, sec=sec, nanosec=nanosec, route=json.loads(M.route_to_json(route))))))
        ego = bundle.ego
        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = "world"
        odometry.child_frame_id = "base_link"
        odometry.pose.pose.position.x, odometry.pose.pose.position.y = float(ego.x), float(ego.y)
        o = odometry.pose.pose.orientation
        o.x, o.y, o.z, o.w = M.yaw_to_quaternion(ego.yaw)
        odometry.twist.twist.linear.x = float(ego.speed)
        odometry.twist.twist.angular.z = float(ego.yaw_rate)
        self.ego.publish(odometry)
        for name, frame in bundle.frames.items():
            if name == skip_camera:
                continue
            image = Image()
            image.header.stamp = stamp
            image.header.frame_id = name
            image.encoding = "mono8"
            if frame.labels is not None:
                image.height, image.width = frame.labels.shape
                image.step = image.width
                image.data = frame.labels.tobytes()
            self.images[name].publish(image)
