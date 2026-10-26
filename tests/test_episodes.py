"""Episode records use only log entries that precede each driver query."""

import math
import struct

import pytest

pytest.importorskip("alpasim_grpc")
from alpasim_grpc.v0.logging_pb2 import LogEntry  # noqa: E402

from metadrive_ais.data.episodes import read_steps  # noqa: E402


def _write(path, entries):
    with open(path, "wb") as f:
        for e in entries:
            data = e.SerializeToString()
            f.write(struct.pack(">L", len(data)) + data)


def _pose(entry_traj, t, x, y, yaw):
    p = entry_traj.poses.add()
    p.timestamp_us = t
    p.pose.vec.x, p.pose.vec.y = x, y
    p.pose.quat.w, p.pose.quat.z = math.cos(yaw / 2), math.sin(yaw / 2)


def _image(t_end, payload):
    e = LogEntry()
    e.driver_camera_image.camera_image.frame_end_us = t_end
    e.driver_camera_image.camera_image.image_bytes = payload
    return e


def _route(lateral):
    e = LogEntry()
    wp = e.route_request.route.waypoints.add()
    wp.x, wp.y = 25.0, lateral
    return e


def _request(t):
    e = LogEntry()
    e.driver_request.time_now_us = t
    return e


def test_no_future_leakage(tmp_path):
    meta = LogEntry()
    meta.rollout_metadata.session_metadata.scene_id = "clipgt-test"
    meta.rollout_metadata.session_metadata.session_uuid = "r1"
    meta.rollout_metadata.session_metadata.start_timestamp_us = 0
    meta.rollout_metadata.force_gt_duration = 300
    ego = LogEntry()
    _pose(ego.driver_ego_trajectory.trajectory, 0, 0.0, 0.0, math.pi / 2)
    _pose(ego.driver_ego_trajectory.trajectory, 100, 0.0, 1.0, math.pi / 2)
    ret = LogEntry()
    _pose(ret.driver_return.trajectory, 100, 0.0, 1.0, math.pi / 2)
    _pose(ret.driver_return.trajectory, 600, 0.0, 6.0, math.pi / 2)  # 5 m straight ahead
    entries = [
        meta,
        _image(90, b"frame-a"),
        _route(+5.0),  # LEFT
        ego,
        _request(100),
        _image(400, b"frame-b"),  # arrives after the first query
        _route(-5.0),  # RIGHT, also after it
        ret,
        _request(500),
    ]
    _write(tmp_path / "rollout.asl", entries)
    meta_out, steps = read_steps(tmp_path / "rollout.asl")
    assert meta_out["scene_id"] == "clipgt-test"
    first, second = steps
    assert first.frame == b"frame-a" and first.command == "LEFT" and first.phase == "forced_replay"
    assert second.frame == b"frame-b" and second.command == "RIGHT" and second.phase == "policy"
    # Issued trajectory in the rig frame: straight ahead along +x.
    assert first.issued_xy[-1] == pytest.approx([5.0, 0.0], abs=1e-3)
    assert first.ego_xyyaw == pytest.approx((0.0, 1.0, math.pi / 2))
