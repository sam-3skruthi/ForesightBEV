"""ROS 2 node: replay a label sequence through real topics, in lockstep with the pipeline.

Decisions come back over DDS and are compared with an in-process run by verify_ros.
``--stall-frame N`` withholds one camera for frame N to exercise the perception watchdog.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import rclpy
from std_msgs.msg import String

from foresightbev.data.sequence import LabelSequence
from foresightbev.integration.ros_common import RELIABLE, FramePublisher, announce_done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--stall-frame", type=int)
    args = parser.parse_args()
    sequence = LabelSequence(args.sequence)
    rclpy.init()
    node = rclpy.create_node("foresight_replay")
    publisher = FramePublisher(node, sequence.calibration)
    done = node.create_publisher(String, "/foresight/sim_status", RELIABLE)
    decisions: list[dict] = []
    node.create_subscription(String, "/foresight/decision", lambda m: decisions.append(json.loads(m.data)), RELIABLE)
    topics: dict[str, int] = {}
    for topic in ("tracks", "detections", "forecast", "risk", "occupancy_state_seen"):
        node.create_subscription(String, f"/foresight/{topic}", lambda m, t=topic: topics.__setitem__(t, topics.get(t, 0) + 1), RELIABLE)
    try:
        deadline = time.monotonic() + 300
        while not publisher.peers_ready():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise TimeoutError("pipeline node did not subscribe")
        start = time.perf_counter()
        for index in range(len(sequence)):
            expected = len(decisions) + 1
            skip = sequence.calibration.names[0] if index == args.stall_frame else None
            publisher.publish(sequence.bundle(index), sequence.route(index), skip)
            deadline = time.monotonic() + args.timeout
            while len(decisions) < expected:
                rclpy.spin_once(node, timeout_sec=0.05)
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no decision for frame {index}")
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        announce_done(node, done)
        result = dict(transport="ros2", kind="replay", sequence=sequence.id, source=sequence.doc["source"], provenance=sequence.doc["provenance"], frames=len(sequence), ros_distro=os.environ.get("ROS_DISTRO"),
                      rmw=rclpy.get_rmw_implementation_identifier(), wall_s=time.perf_counter() - start, topic_message_counts=topics, stall_frame=args.stall_frame, decisions=decisions,
                      scenario=sequence.id, seed=0, collided_with=None, ego_x=0.0)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(f"ros2 replay: {len(decisions)} decisions for {len(sequence)} frames, topics {topics}", flush=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
