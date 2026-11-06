"""Wiring tests against a fake ``carla`` module. They prove nothing about real CARLA."""

import sys
import types
from pathlib import Path

import numpy as np

from foresightbev.contracts import CalibrationBundle, CameraStatus, SemanticClass as C
from foresightbev.data.calibration import load_rig
from foresightbev.integration.carla_backend import carla_frame_to_labels, world_pose_to_ego

RIG = load_rig(Path(__file__).resolve().parents[1] / "configs/rig_cam2bev_1frlr.yaml")


def test_semantic_tag_is_read_from_red_channel():
    raw = np.zeros((1, 3, 4), np.uint8)
    raw[0, :, 2] = (1, 14, 12)
    assert carla_frame_to_labels(raw.tobytes(), 1, 3).tolist() == [[C.ROAD, C.VEHICLE, C.PERSON]]


def test_left_handed_pose_is_converted():
    ego = world_pose_to_ego(1.0, 2.0, 90.0, 3.0, 0.0)
    assert ego.y == -2.0 and np.isclose(ego.yaw, -np.pi / 2)


def fake_carla():
    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Image:
        height, width = 151, 241
        raw_data = np.zeros((151, 241, 4), np.uint8).tobytes()

    class Sensor:
        def listen(self, callback):
            callback(Image())

        def destroy(self):
            pass

    class Actor(Sensor):
        def get_transform(self):
            return Obj(location=Obj(x=0.0, y=0.0), rotation=Obj(yaw=0.0))

        def get_velocity(self):
            return Obj(x=1.0, y=0.0)

    class Blueprint:
        def set_attribute(self, *a):
            pass

    class Library:
        def filter(self, _):
            return [Blueprint()]

        def find(self, _):
            return Blueprint()

    class World:
        def get_settings(self):
            return Obj()

        def apply_settings(self, s):
            self.settings = s

        def get_blueprint_library(self):
            return Library()

        def get_map(self):
            return Obj(get_spawn_points=lambda: [Obj()])

        def spawn_actor(self, blueprint, *a, attach_to=None):
            return Sensor() if attach_to else Actor()

        def tick(self):
            pass

        def get_snapshot(self):
            return Obj(timestamp=Obj(elapsed_seconds=1.5))

    class Client:
        def __init__(self, *a):
            pass

        def set_timeout(self, t):
            pass

        def get_world(self):
            return World()

    module = types.ModuleType("carla")
    module.Client, module.Transform, module.Location, module.Rotation = Client, lambda *a: None, lambda **k: None, lambda **k: None
    return module


def test_adapter_wiring_against_fake_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "carla", fake_carla())
    from foresightbev.integration.carla_backend import CarlaBackend

    rig = CalibrationBundle(tuple(c.scaled(.25) for c in RIG.cameras), "small")
    backend = CarlaBackend(rig)
    bundle = backend.observe()
    assert set(bundle.frames) == set(rig.names) and bundle.timestamp == 1.5
    assert all(f.status == CameraStatus.OK and f.labels.shape == (151, 241) for f in bundle.frames.values())
    assert bundle.ego.speed == 1.0
