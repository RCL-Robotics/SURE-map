# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quick visualization wrapper for GCT predictions using Viser.
"""

import time
import threading
from typing import List, Optional

import numpy as np
import viser
import viser.transforms as tf
from tqdm.auto import tqdm

from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from lingbot_map.vis.sky_segmentation import apply_sky_segmentation


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: Optional[str] = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    This is a simplified wrapper for quick visualization without the full
    PointCloudViewer controls.

    Args:
        pred_dict: Dictionary containing predictions with keys:
            - images: (S, 3, H, W) - Input images
            - world_points: (S, H, W, 3)
            - world_points_conf: (S, H, W)
            - depth: (S, H, W, 1)
            - depth_conf: (S, H, W)
            - extrinsic: (S, 3, 4)
            - intrinsic: (S, 3, 3)
        port: Port number for the viser server
        init_conf_threshold: Initial percentage of low-confidence points to filter out
        use_point_map: Whether to visualize world_points or use depth-based points
        background_mode: Whether to run the server in background thread
        mask_sky: Whether to apply sky segmentation to filter out sky points
        image_folder: Path to the folder containing input images (for sky segmentation)

    Returns:
        viser.ViserServer: The viser server instance
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    shape = world_points.shape
    S: int = shape[0]
    H: int = shape[1]
    W: int = shape[2]

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    # Random sample points if too many
    indices = None
    if points.shape[0] > 6000000:
        print(f"Too many points ({points.shape[0]}), randomly sampling 6M points")
        indices = np.random.choice(points.shape[0], size=6000000, replace=False)
        points = points[indices]
        colors_flat = colors_flat[indices]
        conf_flat = conf_flat[indices]

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices for filtering
    frame_indices = (
        np.repeat(np.arange(S), H * W)[indices]
        if indices is not None
        else np.repeat(np.arange(S), H * W)
    )

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + [str(i) for i in range(S)],
        initial_value="All"
    )

    # Create the main point cloud
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.0005,
        point_shape="circle",
    )

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics, images_: np.ndarray) -> None:
        """Add camera frames and frustums to the scene."""
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(S)):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = tf.SE3.from_matrix(cam2world_3x4)

            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update point cloud based on current GUI selections."""
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add camera frames
    import torch
    if torch.is_tensor(cam_to_world):
        cam_to_world_np = cam_to_world.cpu().numpy()
    else:
        cam_to_world_np = cam_to_world
    visualize_frames(cam_to_world_np, images)

    print("Starting viser server...")
    if background_mode:
        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server
