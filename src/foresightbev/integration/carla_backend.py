"""CARLA backend for the SimBackend protocol. NOT EXECUTED: no CARLA server exists here.

CARLA has no macOS arm64 build, so this adapter has only been run against the fake
``carla`` module in tests/test_carla_adapter.py. That checks wiring and label
mapping, not behaviour against a real server. The semantic tags below are the
CityScapes-based palette of CARLA 0.9.14 and later; check them on your version.

Run on Linux with a CARLA server and the matching ``carla`` Python package:
    sim = CarlaBackend(rig, host="127.0.0.1", port=2000)
    run_episode(sim, PipelineConfig(), "filter")   # after adapting episode metrics
"""

from __future__ import annotations

import queue

import numpy as np

from foresightbev.contracts import (
    CalibrationBundle, CameraFrame, CameraStatus, EgoState, FrameBundle, Provenance, SemanticClass as C,
    TrajectoryCandidate,
)
from foresightbev.risk.trajectories import RouteContext

CARLA_TAG_TO_CLASS = np.zeros(256, np.uint8)
for tag, cls in {1: C.ROAD, 2: C.SIDEWALK, 3: C.STATIC, 4: C.STATIC, 5: C.STATIC, 6: C.STATIC, 7: C.STATIC, 8: C.STATIC,
                 9: C.STATIC, 10: C.OTHER_GROUND, 11: C.SKY, 12: C.PERSON, 13: C.PERSON, 14: C.VEHICLE, 15: C.VEHICLE,
                 16: C.VEHICLE, 17: C.VEHICLE, 18: C.VEHICLE, 19: C.VEHICLE, 20: C.STATIC, 21: C.STATIC, 24: C.ROAD,
                 25: C.OTHER_GROUND, 26: C.STATIC, 27: C.OTHER_GROUND, 28: C.STATIC}.items():
    CARLA_TAG_TO_CLASS[tag] = int(cls)


def carla_frame_to_labels(raw: bytes, height: int, width: int) -> np.ndarray:
    """CARLA BGRA semantic image: the tag is in the red channel."""
    return CARLA_TAG_TO_CLASS[np.frombuffer(raw, np.uint8).reshape(height, width, 4)[:, :, 2]]


def world_pose_to_ego(x: float, y: float, yaw_deg: float, speed: float, t: float) -> EgoState:
    """CARLA is left-handed (y right, yaw clockwise from above); Tavyra is not."""
    return EgoState(t, x, -y, -np.deg2rad(yaw_deg), speed)


class CarlaBackend:
    """Synchronous-mode client. Cameras attach with the rig's pose and intrinsics."""

    def __init__(self, calibration: CalibrationBundle, host: str = "127.0.0.1", port: int = 2000, desired_speed: float = 8.0,
                 fixed_delta: float = 0.05, vehicle_filter: str = "vehicle.tesla.model3"):
        import carla

        self.carla = carla
        self.calibration = calibration
        self.desired_speed = desired_speed
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta
        self.world.apply_settings(settings)
        library = self.world.get_blueprint_library()
        spawn = self.world.get_map().get_spawn_points()[0]
        self.ego = self.world.spawn_actor(library.filter(vehicle_filter)[0], spawn)
        self.queues: dict[str, queue.Queue] = {}
        self.sensors = []
        self.frame = 0
        self.t = 0.0
        for cam in calibration.cameras:
            blueprint = library.find("sensor.camera.semantic_segmentation")
            blueprint.set_attribute("image_size_x", str(cam.width))
            blueprint.set_attribute("image_size_y", str(cam.height))
            blueprint.set_attribute("fov", str(np.rad2deg(2 * np.arctan(cam.width / (2 * cam.K[0, 0])))))
            origin = cam.origin_ego
            forward = cam.T_ego_cam[:3, 2]
            transform = carla.Transform(carla.Location(x=float(origin[0]), y=-float(origin[1]), z=float(origin[2])),
                                        carla.Rotation(yaw=-float(np.rad2deg(np.arctan2(forward[1], forward[0])))))
            sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.ego)
            q: queue.Queue = queue.Queue()
            sensor.listen(q.put)
            self.queues[cam.name] = q
            self.sensors.append(sensor)

    def observe(self) -> FrameBundle:
        self.world.tick()
        snapshot = self.world.get_snapshot()
        self.t = snapshot.timestamp.elapsed_seconds
        transform, velocity = self.ego.get_transform(), self.ego.get_velocity()
        ego = world_pose_to_ego(transform.location.x, transform.location.y, transform.rotation.yaw,
                                float(np.hypot(velocity.x, velocity.y)), self.t)
        frames = {}
        for cam in self.calibration.cameras:
            try:
                image = self.queues[cam.name].get(timeout=2.0)
                labels = carla_frame_to_labels(bytes(image.raw_data), image.height, image.width)
                frames[cam.name] = CameraFrame(cam.name, self.t, labels, CameraStatus.OK)
            except queue.Empty:
                frames[cam.name] = CameraFrame(cam.name, self.t, None, CameraStatus.DROPPED)
        self.frame += 1
        return FrameBundle("carla", self.frame - 1, self.t, frames, self.calibration, ego, Provenance.SIMULATED)

    def route(self) -> RouteContext:
        waypoint = self.world.get_map().get_waypoint(self.ego.get_location())
        location, yaw = waypoint.transform.location, -np.deg2rad(waypoint.transform.rotation.yaw)
        here = self.ego.get_transform()
        ego_yaw = -np.deg2rad(here.rotation.yaw)
        dx, dy = here.location.x - location.x, -(here.location.y - location.y)
        lateral = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return RouteContext(self.desired_speed, float(lateral), float(ego_yaw - yaw), (0.0,), 0.0)

    def apply(self, trajectory: TrajectoryCandidate, duration: float) -> None:
        target = np.array([np.interp(duration, trajectory.times, trajectory.poses[:, j]) for j in range(4)])
        speed = float(self.ego.get_velocity().length())
        error = float(target[3]) - speed
        control = self.carla.VehicleControl(throttle=float(np.clip(error * 0.5, 0, 1)), brake=float(np.clip(-error * 0.4, 0, 1)),
                                            steer=float(np.clip(-target[2], -1, 1)))
        self.ego.apply_control(control)

    def status(self) -> dict:
        return dict(t=self.t, backend="carla")

    def close(self) -> None:
        for sensor in self.sensors:
            sensor.destroy()
        self.ego.destroy()
