"""ROS 2 node: built-in simulator in lockstep. Run inside the pixi environment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from std_msgs.msg import String

from foresightbev.evaluation.experiments import load_experiment, noise_from, scaled_rig
from foresightbev.integration import ros_messages as M
from foresightbev.integration.ros_common import RELIABLE, FramePublisher, announce_done
from foresightbev.sim.backend import BuiltinSim
from foresightbev.sim.closed_loop import CONTROL_DT
from foresightbev.sim.scenarios import build_scenario

STARTUP_TIMEOUT_S = 300.0


class SimNode:
    def __init__(self, args):
        root = Path(args.root)
        experiment = load_experiment(root / args.config)
        scenario = build_scenario(args.scenario, args.seed)
        self.rig = scaled_rig(experiment, root)
        self.sim = BuiltinSim(scenario, self.rig, noise_from(experiment), seed=args.seed)
        self.scenario = scenario
        self.limit = scenario.duration_s
        self.node = rclpy.create_node("foresight_sim")
        self.frames = FramePublisher(self.node, self.rig)
        self.done = self.node.create_publisher(String, "/foresight/sim_status", RELIABLE)
        self.pending = None
        self.decisions = []
        self.node.create_subscription(String, "/foresight/trajectory_timed", self.on_trajectory, RELIABLE)
        self.node.create_subscription(String, "/foresight/decision", lambda m: self.decisions.append(json.loads(m.data)), RELIABLE)

    def on_trajectory(self, message: String) -> None:
        self.pending = M.trajectory_from_json(message.data)

    def publish_frame(self) -> int:
        bundle = self.sim.observe()
        self.frames.publish(bundle, self.sim.route())
        return bundle.frame_index

    def wait_for_peers(self, timeout: float) -> None:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.frames.peers_ready():
                return
            rclpy.spin_once(self.node, timeout_sec=0.1)
        raise TimeoutError("pipeline node did not subscribe")

    def run(self, timeout: float) -> dict:
        self.wait_for_peers(STARTUP_TIMEOUT_S)
        scenario = self.scenario
        start = time.perf_counter()
        frames = 0
        while self.sim.t < self.limit - 1e-6:
            frame = self.publish_frame()
            self.pending = None
            deadline = time.monotonic() + timeout
            while self.pending is None or self.pending[0] != frame:
                rclpy.spin_once(self.node, timeout_sec=0.05)
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no trajectory for frame {frame}")
            self.sim.apply(self.pending[1], CONTROL_DT)
            frames += 1
            if self.sim.collided_with is not None or scenario.ego.x >= scenario.goal_x:
                break
        end = time.monotonic() + 1.0
        while time.monotonic() < end and len(self.decisions) < frames:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        announce_done(self.node, self.done)
        return dict(transport="ros2", ros_distro=None, scenario=scenario.name, seed=scenario.seed, frames=frames,
                    collided_with=self.sim.collided_with, ego_x=float(scenario.ego.x), t=self.sim.t,
                    wall_s=time.perf_counter() - start, decisions=self.decisions[:frames])


def main() -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/experiment_default.yaml")
    parser.add_argument("--scenario", default="occluded_pedestrian")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rclpy.init()
    try:
        node = SimNode(args)
        result = node.run(args.timeout)
        result["ros_distro"] = os.environ.get("ROS_DISTRO")
        result["rmw"] = rclpy.get_rmw_implementation_identifier()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(f"ros2 closed loop: {result['frames']} frames, collided={result['collided_with']}, x={result['ego_x']:.1f}, "
              f"wall={result['wall_s']:.1f}s", flush=True)
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
