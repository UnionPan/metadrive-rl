"""AlpaSim sensorsim gRPC service backed by the gsplat renderer.

A drop-in replacement for NVIDIA's NRE renderer on GPUs NRE does not support
(e.g. Turing). Implements the RPCs the AlpaSim runtime uses for camera
rendering: version, scene and camera discovery, ego masks (none), and single
or batched RGB rendering. Lidar and aggregated rendering are not implemented.

Usage:
    python -m metadrive_ais.render.server --port 46100 \\
        --artifact-glob '/data/nre-artifacts/all-usdzs/*.usdz'
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import threading
import zipfile
from collections import OrderedDict
from concurrent import futures

import grpc
import numpy as np
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc
from PIL import Image
from scipy.spatial.transform import Rotation

import metadrive_ais

from .rasterizer import GsplatRenderer, ObjectPose
from .scene import CameraCalibration, NuRecScene

logger = logging.getLogger(__name__)


def pose_to_matrix(pose: common_pb2.Pose) -> np.ndarray:
    T = np.eye(4)
    q = pose.quat
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = (pose.vec.x, pose.vec.y, pose.vec.z)
    return T


def matrix_to_pose(T: np.ndarray) -> common_pb2.Pose:
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
        quat=common_pb2.Quat(w=w, x=x, y=y, z=z),
    )


def midpoints(pairs) -> np.ndarray:
    """Pose halfway (slerp) between each pair's start and end, (K, 4, 4)."""

    def stack(poses):
        q = np.array([[p.quat.x, p.quat.y, p.quat.z, p.quat.w] for p in poses])
        v = np.array([[p.vec.x, p.vec.y, p.vec.z] for p in poses])
        return Rotation.from_quat(q), v

    r0, v0 = stack([p.start_pose for p in pairs])
    r1, v1 = stack([p.end_pose for p in pairs])
    T = np.tile(np.eye(4), (len(pairs), 1, 1))
    T[:, :3, :3] = (r0 * Rotation.from_rotvec((r0.inv() * r1).as_rotvec() * 0.5)).as_matrix()
    T[:, :3, 3] = 0.5 * (v0 + v1)
    return T


def camera_spec(calib: CameraCalibration) -> sensorsim_pb2.CameraSpec:
    w, h = calib.resolution_wh
    return sensorsim_pb2.CameraSpec(
        logical_id=calib.logical_id,
        resolution_w=w,
        resolution_h=h,
        shutter_type=sensorsim_pb2.ShutterType.Value(calib.shutter_type),
        ftheta_param=sensorsim_pb2.FthetaCameraParam(
            principal_point_x=calib.principal_point[0],
            principal_point_y=calib.principal_point[1],
            reference_poly=sensorsim_pb2.FthetaCameraParam.PolynomialType.Value(
                "PIXELDIST_TO_ANGLE"
            ),
            pixeldist_to_angle_poly=calib.pixeldist_to_angle_poly,
            angle_to_pixeldist_poly=calib.angle_to_pixeldist_poly,
            max_angle=calib.max_angle,
            linear_cde=sensorsim_pb2.LinearCde(
                linear_c=calib.linear_cde[0],
                linear_d=calib.linear_cde[1],
                linear_e=calib.linear_cde[2],
            ),
        ),
    )


def calibration_from_spec(
    spec: sensorsim_pb2.CameraSpec, T_sensor_rig: np.ndarray
) -> CameraCalibration:
    kind = spec.WhichOneof("camera_param")
    if kind == "opencv_pinhole_param":
        p = spec.opencv_pinhole_param
        return CameraCalibration(
            logical_id=spec.logical_id,
            T_sensor_rig=T_sensor_rig,
            resolution_wh=(spec.resolution_w, spec.resolution_h),
            principal_point=(p.principal_point_x, p.principal_point_y),
            shutter_type=sensorsim_pb2.ShutterType.Name(spec.shutter_type),
            camera_model="pinhole",
            focal_length=(p.focal_length_x, p.focal_length_y),
            radial_coeffs=tuple(p.radial_coeffs),
            tangential_coeffs=tuple(p.tangential_coeffs),
            thin_prism_coeffs=tuple(p.thin_prism_coeffs),
        )
    if kind != "ftheta_param":
        raise ValueError(f"camera {spec.logical_id}: unsupported camera model {kind}")
    p = spec.ftheta_param
    if p.reference_poly != sensorsim_pb2.FthetaCameraParam.PolynomialType.Value(
        "PIXELDIST_TO_ANGLE"
    ):
        raise ValueError(f"camera {spec.logical_id}: only PIXELDIST_TO_ANGLE f-theta is supported")
    return CameraCalibration(
        logical_id=spec.logical_id,
        T_sensor_rig=T_sensor_rig,
        resolution_wh=(spec.resolution_w, spec.resolution_h),
        principal_point=(p.principal_point_x, p.principal_point_y),
        pixeldist_to_angle_poly=tuple(p.pixeldist_to_angle_poly),
        angle_to_pixeldist_poly=tuple(p.angle_to_pixeldist_poly),
        max_angle=p.max_angle,
        linear_cde=(p.linear_cde.linear_c, p.linear_cde.linear_d, p.linear_cde.linear_e),
        shutter_type=sensorsim_pb2.ShutterType.Name(spec.shutter_type),
    )


def encode(image: np.ndarray, image_format: int, quality: float) -> bytes:
    buffer = io.BytesIO()
    if image_format == sensorsim_pb2.ImageFormat.PNG:
        Image.fromarray(image).save(buffer, format="PNG")
    elif image_format == sensorsim_pb2.ImageFormat.JPEG:
        Image.fromarray(image).save(buffer, format="JPEG", quality=int(quality) or 95)
    else:
        raise ValueError(f"unsupported image format {sensorsim_pb2.ImageFormat.Name(image_format)}")
    return buffer.getvalue()


def scene_id_of(usdz: str) -> str:
    with zipfile.ZipFile(usdz) as z:
        return json.loads(z.read("rig_trajectories.json"))["rig_trajectories"][0]["sequence_id"]


class GsplatSensorsim(sensorsim_pb2_grpc.SensorsimServiceServicer):
    def __init__(self, artifact_glob: str, device: str, cache_size: int, min_opacity: float):
        self.paths = {scene_id_of(p): p for p in sorted(glob.glob(artifact_glob, recursive=True))}
        if not self.paths:
            raise FileNotFoundError(f"no scenes match {artifact_glob}")
        self.device, self.cache_size, self.min_opacity = device, cache_size, min_opacity
        self._renderers: OrderedDict[str, GsplatRenderer] = OrderedDict()
        self._lock = threading.Lock()  # one GPU, one render at a time
        logger.info("Scenes available: %s", sorted(self.paths))

    def _renderer(self, scene_id: str) -> GsplatRenderer:
        if scene_id in self._renderers:
            self._renderers.move_to_end(scene_id)
            return self._renderers[scene_id]
        if scene_id not in self.paths:
            raise KeyError(f"unknown scene {scene_id}")
        if len(self._renderers) >= self.cache_size:
            while len(self._renderers) >= self.cache_size:
                self._renderers.popitem(last=False)
            # Return the evicted scenes' memory to the driver: renderers share
            # GPUs with physics, CATK and drivers, which otherwise run out.
            torch.cuda.empty_cache()
        logger.info("Loading scene %s", scene_id)
        renderer = GsplatRenderer(
            NuRecScene(self.paths[scene_id]), self.device, min_opacity=self.min_opacity
        )
        self._renderers[scene_id] = renderer
        return renderer

    def _render(self, request: sensorsim_pb2.RGBRenderRequest) -> bytes:
        with self._lock:
            renderer = self._renderer(request.scene_id)
            calib = calibration_from_spec(request.camera_intrinsics, np.eye(4))
            dyn = request.dynamic_objects
            objects = (
                [
                    ObjectPose(o.track_id, T)
                    for o, T in zip(dyn, midpoints([o.pose_pair for o in dyn]), strict=True)
                ]
                if dyn
                else []
            )
            image = renderer.render(
                calib,
                request.resolution_w or calib.resolution_wh[0],
                request.resolution_h or calib.resolution_wh[1],
                pose_to_matrix(request.sensor_pose.start_pose),
                pose_to_matrix(request.sensor_pose.end_pose),
                objects,
            )
        return encode(image, request.image_format, request.image_quality)

    # --- RPCs -------------------------------------------------------------
    def get_version(self, request, context):
        return common_pb2.VersionId(
            version_id=f"metadrive-gsplat-{metadrive_ais.__version__}",
            git_hash="",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def get_available_scenes(self, request, context):
        return common_pb2.AvailableScenesReturn(scene_ids=sorted(self.paths))

    def get_available_cameras(self, request, context):
        with self._lock:
            scene = self._renderer(request.scene_id).scene
        return sensorsim_pb2.AvailableCamerasReturn(
            available_cameras=[
                sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
                    intrinsics=camera_spec(calib),
                    rig_to_camera=matrix_to_pose(calib.T_sensor_rig),
                    logical_id=calib.logical_id,
                )
                for calib in scene.cameras.values()
            ]
        )

    def get_available_ego_masks(self, request, context):
        return sensorsim_pb2.AvailableEgoMasksReturn()

    def render_rgb(self, request, context):
        try:
            return sensorsim_pb2.RGBRenderReturn(image_bytes=self._render(request))
        except (KeyError, ValueError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    def batch_render_rgb(self, request, context):
        items = []
        for item in request.items:
            try:
                result = sensorsim_pb2.RGBRenderReturn(image_bytes=self._render(item.request))
                items.append(
                    sensorsim_pb2.BatchRGBRenderReturnItem(
                        camera_name=item.camera_name, result=result, success=True
                    )
                )
            except (KeyError, ValueError) as exc:
                items.append(
                    sensorsim_pb2.BatchRGBRenderReturnItem(
                        camera_name=item.camera_name, success=False, error_message=str(exc)
                    )
                )
        return sensorsim_pb2.BatchRGBRenderReturn(items=items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=46100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--artifact-glob", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-size", type=int, default=2, help="scenes kept on the GPU")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.01,
        help="prune near-transparent static Gaussians (0.01: -41%% Gaussians, -12%% render "
        "time, no PSNR change on 24 validation scenes)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    servicer = GsplatSensorsim(args.artifact_glob, args.device, args.cache_size, args.min_opacity)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            ("grpc.max_send_message_length", 64 << 20),
            ("grpc.max_receive_message_length", 64 << 20),
        ],
    )
    sensorsim_pb2_grpc.add_SensorsimServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logger.info("gsplat sensorsim serving on %s:%d", args.host, args.port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
