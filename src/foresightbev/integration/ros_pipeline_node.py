"""ROS 2 node: the Tavyra pipeline. All inputs arrive as ROS messages.

Publishes every stage output. With ``--record`` it also writes a normal run directory,
so ROS output is served by the backend like any other run. A watchdog publishes a
PERCEPTION_MISSING fallback when an expected frame does not complete in time, because
silence must not read as a clear road.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
from sensor_msgs.msg import Image
from std_msgs.msg import String

from foresightbev.config import load_pipeline_config
from foresightbev.contracts import CameraFrame, CameraStatus, EgoState, FrameBundle, Provenance
from foresightbev.data.sequence import LabelSequence
from foresightbev.errors import ForesightError
from foresightbev.evaluation.experiments import build_mode, load_experiment, scaled_rig
from foresightbev.integration import ros_messages as M
from foresightbev.integration.ros_common import RELIABLE
from foresightbev.pipeline import ForesightPipeline
from foresightbev.replay import build_pipeline
from foresightbev.recording import RunRecorder, provenance_table
from foresightbev.risk.trajectories import RouteContext

logger = logging.getLogger(__name__)


class PipelineNode:
    def __init__(self, args):
        root = Path(args.root)
        if args.sequence:
            sequence = LabelSequence(args.sequence)
            self.rig = sequence.calibration
            self.source = Provenance(sequence.doc["provenance"])
            config = load_pipeline_config(args.pipeline_config)
            table = provenance_table(sequence.doc["provenance"], f"{sequence.doc['source']} over ROS 2 topics.", sequence.doc.get("segmenter"), detector=config.detector)
        else:
            experiment = load_experiment(root / args.config)
            self.rig = scaled_rig(experiment, root)
            self.source = Provenance.SIMULATED
            config, _ = build_mode(experiment["modes"][args.mode], experiment.get("pipeline"))
            table = provenance_table("simulated", "Built-in simulator over ROS 2 topics.", closed_loop="ROS 2 lockstep with the built-in simulator.")
        config.seed = args.seed
        self.pipeline = build_pipeline(sequence, config) if args.sequence else ForesightPipeline(self.rig, config)
        self.config = config
        self.watchdog_s = args.watchdog
        self.recorder = None
        if args.record:
            directory = Path(args.record)
            if directory.exists():
                shutil.rmtree(directory)
            self.recorder = RunRecorder(directory, dict(kind="ros2", title=directory.name, scenario=directory.name, seed=args.seed, mode="ros2", input="ROS 2 topics",
                                                        caveat="Recorded by the ROS 2 pipeline node from messages received over DDS.", provenance=table), config.grid.bev)
        self.node = rclpy.create_node("foresight_pipeline")
        self.partial: dict[int, dict] = {}
        self.finished = False
        self.last = None
        self.waiting_since: float | None = None
        self.last_activity: float | None = None
        for cam in self.rig.cameras:
            self.node.create_subscription(Image, f"/foresight/cameras/{cam.name}/labels", lambda m, name=cam.name: self.on_image(name, m), RELIABLE)
        self.node.create_subscription(Odometry, "/foresight/ego", self.on_ego, RELIABLE)
        self.node.create_subscription(String, "/foresight/route", self.on_route, RELIABLE)
        self.node.create_subscription(String, "/foresight/sim_status", self.on_status, RELIABLE)
        self.ready_pub = self.node.create_publisher(String, "/foresight/pipeline_ready", RELIABLE)
        self.grid_pub = self.node.create_publisher(OccupancyGrid, "/foresight/occupancy", RELIABLE)
        self.state_pub = self.node.create_publisher(Image, "/foresight/occupancy_state", RELIABLE)
        self.path_pub = self.node.create_publisher(PathMsg, "/foresight/trajectory", RELIABLE)
        self.json_pubs = {name: self.node.create_publisher(String, f"/foresight/{name}", RELIABLE) for name in ("decision", "trajectory_timed", "tracks", "detections", "forecast", "risk")}

    def announce_ready(self) -> None:
        """Say so only when every reader has matched a writer, until data flows."""
        if self.last_activity is not None:
            return
        topics = [f"/foresight/cameras/{c.name}/labels" for c in self.rig.cameras] + ["/foresight/ego", "/foresight/route"]
        if all(self.node.count_publishers(topic) > 0 for topic in topics):
            self.ready_pub.publish(String(data="ready"))

    def idle_too_long(self, limit: float) -> bool:
        """Stop if the source vanished without an end signal, so nothing waits forever."""
        return self.last_activity is not None and time.monotonic() - self.last_activity > limit

    def _slot(self, key: int) -> dict:
        self.last_activity = time.monotonic()
        if self.waiting_since is None:
            self.waiting_since = time.monotonic()
        return self.partial.setdefault(key, dict(images={}))

    def on_image(self, name: str, message: Image) -> None:
        key = M.stamp_key(message.header.stamp.sec, message.header.stamp.nanosec)
        labels = None
        if message.height:
            labels = np.frombuffer(bytes(message.data), np.uint8).reshape(message.height, message.width)
        self._slot(key)["images"][name] = labels
        self.try_process(key)

    def on_ego(self, message: Odometry) -> None:
        stamp = message.header.stamp
        key = M.stamp_key(stamp.sec, stamp.nanosec)
        q, p = message.pose.pose.orientation, message.pose.pose.position
        self._slot(key)["ego"] = EgoState(M.from_stamp(stamp.sec, stamp.nanosec), p.x, p.y, M.quaternion_to_yaw(q.x, q.y, q.z, q.w), message.twist.twist.linear.x, message.twist.twist.angular.z)
        self.try_process(key)

    def on_route(self, message: String) -> None:
        d = json.loads(message.data)
        key = M.stamp_key(d["sec"], d["nanosec"])
        slot = self._slot(key)
        slot["route"] = M.route_from_json(json.dumps(d["route"]))
        slot["frame"] = d["frame"]
        self.try_process(key)

    def on_status(self, message: String) -> None:
        self.finished = True

    def try_process(self, key: int) -> None:
        slot = self.partial.get(key)
        if slot is None or "ego" not in slot or "route" not in slot or len(slot["images"]) < len(self.rig.cameras):
            return
        del self.partial[key]
        self.waiting_since = time.monotonic() if self.partial else None
        ego = slot["ego"]
        images = slot["images"]
        frames = {c.name: CameraFrame(c.name, ego.timestamp, images[c.name], CameraStatus.OK if images[c.name] is not None else CameraStatus.DROPPED) for c in self.rig.cameras}
        bundle = FrameBundle("ros2", slot["frame"], ego.timestamp, frames, self.rig, ego, self.source)
        try:
            result = self.pipeline.step(bundle, slot["route"])
        except ForesightError as error:
            logger.error("rejected frame %s: %s", slot["frame"], error)
            self.publish_missing(ego, slot["route"], slot["frame"])
            return
        self.last = (ego, slot["route"])
        self.publish(result, ego)
        if self.recorder is not None:
            self.recorder.add(result)

    def check_watchdog(self) -> None:
        """An incomplete frame older than the limit becomes a safe fallback decision."""
        if self.waiting_since is None or time.monotonic() - self.waiting_since < self.watchdog_s or not self.partial:
            return
        key = min(self.partial)
        slot = self.partial.pop(key)
        self.waiting_since = time.monotonic() if self.partial else None
        ego = slot.get("ego") or (self.last[0] if self.last else None)
        route = slot.get("route") or (self.last[1] if self.last else None)
        if ego is None or route is None:
            # Nothing is known about the vehicle, but a stop is still a valid command.
            logger.error("incomplete frame and no ego state ever received; commanding a stop")
            ego, route = ego or EgoState(0.0, 0.0, 0.0, 0.0), route or RouteContext()
        logger.warning("frame %s incomplete after %.1f s: %d of %d cameras", slot.get("frame"), self.watchdog_s, len(slot["images"]), len(self.rig.cameras))
        self.publish_missing(ego, route, slot.get("frame", -1))

    def publish_missing(self, ego: EgoState, route, frame: int) -> None:
        decision = self.pipeline.decide_without_perception(ego, route)
        self.json_pubs["decision"].publish(String(data=M.decision_to_json(decision, frame, ego.timestamp)))
        self.json_pubs["trajectory_timed"].publish(String(data=M.trajectory_to_json(decision.selected, frame)))

    def publish(self, result, ego: EgoState) -> None:
        grid = result.occupancy.grid.bev
        sec, nanosec = M.to_stamp(ego.timestamp)
        stamp = rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = "base_link"
        message.info.resolution = float(grid.resolution)
        message.info.width, message.info.height = grid.nx, grid.ny
        message.info.origin.position.x, message.info.origin.position.y = float(grid.x_min), float(grid.y_min)
        message.info.origin.orientation.w = 1.0
        message.data = M.occupancy_grid_data(result.occupancy.bev_state).tolist()
        self.grid_pub.publish(message)
        state = Image()
        state.header = message.header
        state.height, state.width = result.occupancy.bev_state.shape
        state.step, state.encoding = state.width, "mono8"
        state.data = result.occupancy.bev_state.tobytes()
        self.state_pub.publish(state)
        decision = result.decision
        path = PathMsg()
        path.header = message.header
        for t, pose in zip(decision.selected.times[::4], decision.selected.poses[::4]):
            item = PoseStamped()
            item.header.stamp.sec, item.header.stamp.nanosec = M.to_stamp(ego.timestamp + t)
            item.header.frame_id = "base_link"
            item.pose.position.x, item.pose.position.y = float(pose[0]), float(pose[1])
            q = item.pose.orientation
            q.x, q.y, q.z, q.w = M.yaw_to_quaternion(float(pose[2]))
            path.poses.append(item)
        self.path_pub.publish(path)
        frame = result.bundle.frame_index
        for name, text in (("detections", M.detections_to_json(result)), ("tracks", M.tracks_to_json(result)), ("forecast", M.forecast_to_json(result)), ("risk", M.risk_to_json(result))):
            self.json_pubs[name].publish(String(data=text))
        self.json_pubs["decision"].publish(String(data=M.decision_to_json(decision, frame, result.bundle.timestamp, M.state_hash(result.occupancy.bev_state))))
        self.json_pubs["trajectory_timed"].publish(String(data=M.trajectory_to_json(decision.selected, frame)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/experiment_default.yaml")
    parser.add_argument("--mode", default="filter")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--sequence", help="take calibration from this label sequence")
    parser.add_argument("--pipeline-config")
    parser.add_argument("--record", help="write a run directory")
    parser.add_argument("--watchdog", type=float, default=5.0)
    parser.add_argument("--idle-exit", type=float, default=60.0, help="exit after this long without input once frames have arrived")
    args = parser.parse_args()
    rclpy.init()
    node = PipelineNode(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node.node, timeout_sec=0.1)
            node.announce_ready()
            node.check_watchdog()
            if node.idle_too_long(args.idle_exit):
                logger.error("no input for %.0f s after the last frame; exiting", args.idle_exit)
                break
    except ExternalShutdownException:
        logger.warning("pipeline node shut down externally")
    finally:
        if node.recorder is not None:
            node.recorder.finish(dict(frames=node.recorder.count))
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
