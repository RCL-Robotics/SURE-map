"""SURE-Map streaming reconstruction with geometric uncertainty filtering."""

import logging
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.method.base import BaseMethod
from benchmark.core.loader import BSSLoader
from benchmark.io.image import save_exr


# Mirrors lingbot-map/demo.py:413-421 — above this frame count, the KV cache
# grows unbounded, so auto-bump the keyframe interval.
_AUTO_KEYFRAME_THRESHOLD = 320


def _resolve_keyframe_interval(cfg_val, num_frames: int) -> int:
    """Resolve a raw config value into a concrete keyframe interval.

    ``None``, ``0``, or the string ``"auto"`` triggers auto-selection:
    ``1`` when ``num_frames <= 320`` else ``ceil(num_frames / 320)``.
    An explicit positive int is returned as-is.
    """
    if cfg_val is None or cfg_val == 0 or (isinstance(cfg_val, str) and cfg_val.lower() == "auto"):
        if num_frames <= _AUTO_KEYFRAME_THRESHOLD:
            return 1
        return (num_frames + _AUTO_KEYFRAME_THRESHOLD - 1) // _AUTO_KEYFRAME_THRESHOLD
    return int(cfg_val)


class SureMapMethod(BaseMethod):
    """
    SURE-Map adapter for benchmark evaluation.

    Supports streaming and windowed inference modes via the upstream
    ``GCTStream`` model exposed by the ``lingbot_map`` package.
    """

    def __init__(
        self,
        checkpoint: str = None,
        device: str = 'cuda',
        mode: str = 'streaming',
        use_amp: bool = True,
        use_sdpa: bool = False,
        image_size: int = 518,
        patch_size: int = 14,
        enable_3d_rope: bool = True,
        num_scale_frames: int = 8,
        max_frame_num: int = 1024,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        window_size: int = 64,
        overlap_size: Optional[int] = None,
        keyframe_interval: Any = "auto",
        flow_threshold: float = 0.0,
        max_non_keyframe_gap: int = 30,
        sigma_ckpt: Optional[str] = None,
        sigma_confidence_eps: float = 1e-6,
        align: int = 14,
        area_budget: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(
            align=align,
            area_budget=area_budget,
            logger=logger,
        )

        self.checkpoint = checkpoint
        self.device = device
        self.mode = mode
        self.use_amp = use_amp
        self.use_sdpa = use_sdpa
        self.image_size = image_size
        self.patch_size = patch_size
        self.enable_3d_rope = enable_3d_rope
        self.num_scale_frames = num_scale_frames
        self.max_frame_num = max_frame_num
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.keyframe_interval = keyframe_interval
        self.flow_threshold = flow_threshold
        self.max_non_keyframe_gap = max_non_keyframe_gap
        self.sigma_ckpt = sigma_ckpt
        self.sigma_confidence_eps = float(sigma_confidence_eps)
        self.sigma_head = None

        if self.mode not in ('streaming', 'windowed'):
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'streaming' or 'windowed'")

        self._load_model()
        self._load_sigma_head()

    def _load_model(self):
        """Load LingbotMap (GCTStream) model from checkpoint."""
        if self.mode == 'windowed':
            from lingbot_map.models.gct_stream_window import GCTStream
        else:
            from lingbot_map.models.gct_stream import GCTStream

        print(f"  → Building LingbotMap model (mode: {self.mode})")
        self.model = GCTStream(
            img_size=self.image_size,
            patch_size=self.patch_size,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.use_sdpa,
        )

        if self.checkpoint:
            print(f"  → Loading checkpoint: {self.checkpoint}")
            ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"    Missing keys: {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
            print("    Checkpoint loaded.")

        self.model = self.model.to(self.device).eval()

    def _load_sigma_head(self):
        """Optionally load the trained FlowSigmaHead used as point confidence."""
        if not self.sigma_ckpt:
            return

        repo_root = Path(__file__).resolve().parents[2]
        training_dir = repo_root / "training"
        if str(training_dir) not in sys.path:
            sys.path.insert(0, str(training_dir))

        from heads.flow_sigma_head import FlowSigmaHead

        print(f"  → Loading sigma head: {self.sigma_ckpt}")
        head = FlowSigmaHead(
            dim_in=2 * self.model.embed_dim,
            patch_size=self.patch_size,
            num_heads=8,
            log_sigma_min=-2.0,
            log_sigma_init=0.0,
        )
        sigma_ckpt = torch.load(self.sigma_ckpt, map_location="cpu", weights_only=False)
        head.load_state_dict(sigma_ckpt["sigma_head"], strict=True)
        self.sigma_head = head.to(self.device).eval()

    def _run_inference_with_sigma(self, images, dtype):
        """Run streaming inference and replace depth confidence with uncertainty confidence.

        The sigma head predicts adjacent-frame image-space uncertainty. BSS confidence
        keeps larger values under percentile filtering, so we store 1 / sigma_uv.
        """
        if self.mode != 'streaming':
            self.logger.warning("Sigma confidence is only implemented for streaming mode; using model depth_conf.")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                return self.model.inference_windowed(
                    images,
                    window_size=self.window_size,
                    overlap_size=self.overlap_size,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=self.keyframe_interval,
                    flow_threshold=self.flow_threshold,
                    max_non_keyframe_gap=self.max_non_keyframe_gap,
                    output_device=torch.device("cpu"),
                )

        sigma_conf_frames: List[Optional[torch.Tensor]] = []
        sigma_uv_frames: List[Optional[torch.Tensor]] = []
        prev_agg: Optional[List[torch.Tensor]] = None

        def _append_sigma_from_hook(_module, inputs, output):
            nonlocal prev_agg
            agg = output[0] if isinstance(output, (tuple, list)) else output
            patch_start = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else 6
            block_images = inputs[0]
            agg_f = [t.detach().float() for t in agg]
            block_images_f = block_images.detach().float()
            block_len = agg_f[0].shape[1]

            for j in range(block_len):
                cur_agg = [t[:, j:j + 1] for t in agg_f]
                if prev_agg is None:
                    sigma_conf_frames.append(None)
                    sigma_uv_frames.append(None)
                else:
                    lsu, lsv = self.sigma_head(
                        cur_agg,
                        prev_agg,
                        block_images_f[:, j:j + 1],
                        patch_start,
                    )
                    sigma_u = torch.sqrt(torch.exp(lsu[:, 0]).clamp_min(1e-12))
                    sigma_v = torch.sqrt(torch.exp(lsv[:, 0]).clamp_min(1e-12))
                    sigma_uv = torch.sqrt(sigma_u.square() + sigma_v.square()).clamp_min(self.sigma_confidence_eps)
                    sigma_uv_cpu = sigma_uv.detach().cpu()
                    sigma_uv_frames.append(sigma_uv_cpu)
                    sigma_conf_frames.append(1.0 / sigma_uv_cpu)
                prev_agg = [t.detach() for t in cur_agg]

        handle = self.model.aggregator.register_forward_hook(_append_sigma_from_hook)
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                num_frames = images.shape[0]
                keyframe_interval = _resolve_keyframe_interval(self.keyframe_interval, num_frames)
                if keyframe_interval != self.keyframe_interval:
                    print(
                        f"  → Auto-selected keyframe_interval={keyframe_interval} "
                        f"(num_frames={num_frames}, raw={self.keyframe_interval!r})"
                    )
                predictions = self.model.inference_streaming(
                    images,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=torch.device("cpu"),
                )
        finally:
            handle.remove()

        num_pred = predictions["pose_enc"].shape[1]
        sigma_conf_frames = sigma_conf_frames[:num_pred]
        sigma_uv_frames = sigma_uv_frames[:num_pred]
        valid = [x for x in sigma_conf_frames if x is not None]
        if valid:
            neutral = torch.cat([x.reshape(-1) for x in valid]).median().item()
            template = valid[0]
        else:
            _, _, h, w = predictions["depth"].shape[:4]
            neutral = 1.0
            template = torch.ones((1, h, w), dtype=torch.float32)
        while len(sigma_conf_frames) < num_pred:
            sigma_conf_frames.append(None)
        while len(sigma_uv_frames) < num_pred:
            sigma_uv_frames.append(None)
        filled = [
            x if x is not None else torch.full_like(template, neutral)
            for x in sigma_conf_frames
        ]
        sigma_neutral = 1.0 / max(neutral, self.sigma_confidence_eps)
        sigma_filled = [
            x if x is not None else torch.full_like(template, sigma_neutral)
            for x in sigma_uv_frames
        ]
        predictions["depth_conf"] = torch.stack(filled, dim=1).float()
        predictions["sigma_uv"] = torch.stack(sigma_filled, dim=1).float()
        print(f"  → Saved sigma-derived confidence from {len(valid)} adjacent-frame sigma maps")
        return predictions

    def _prepare_images(self, rgb_list):
        """Convert list of HxWx3 uint8 numpy arrays to [S, 3, H, W] tensor in [0, 1]."""
        from torchvision import transforms as TF

        to_tensor = TF.ToTensor()
        images = torch.stack([to_tensor(rgb) for rgb in rgb_list])
        return images.to(self.device)

    def _run_inference(self, images):
        """Run SURE-Map inference and return raw predictions dict."""
        if self.use_amp:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        print(f"  → Running {self.mode} inference (dtype: {dtype})")

        if self.sigma_head is not None:
            return self._run_inference_with_sigma(images, dtype)

        num_frames = images.shape[0]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            if self.mode == 'streaming':
                keyframe_interval = _resolve_keyframe_interval(self.keyframe_interval, num_frames)
                if keyframe_interval != self.keyframe_interval:
                    print(
                        f"  → Auto-selected keyframe_interval={keyframe_interval} "
                        f"(num_frames={num_frames}, raw={self.keyframe_interval!r})"
                    )
                predictions = self.model.inference_streaming(
                    images,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=torch.device("cpu"),
                )
            else:
                predictions = self.model.inference_windowed(
                    images,
                    window_size=self.window_size,
                    overlap_size=self.overlap_size,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=self.keyframe_interval,
                    flow_threshold=self.flow_threshold,
                    max_non_keyframe_gap=self.max_non_keyframe_gap,
                    output_device=torch.device("cpu"),
                )

        return predictions

    def _process_outputs(self, predictions, image_shape):
        """Convert model predictions to benchmark output format.

        Args:
            predictions: Raw model outputs with 'pose_enc', 'depth', 'depth_conf', etc.
            image_shape: (H, W) of the processed images.

        Returns:
            Tuple of (rgb_list, depth_list, pose_list, intrinsics_list, confidence_list, sigma_list)
        """
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

        # Decode pose encoding to extrinsic + intrinsic
        # pose_encoding_to_extri_intri() output is C2W directly (no inverse needed)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], image_shape
        )

        extrinsic = extrinsic.float().cpu().numpy().squeeze(0)  # [S, 3, 4]
        intrinsic = intrinsic.float().cpu().numpy().squeeze(0)  # [S, 3, 3]
        depth = predictions["depth"].float().cpu().numpy().squeeze(0)  # [S, H, W, 1]

        # Extract processed images
        if "images" in predictions:
            images = predictions["images"].float().cpu().numpy().squeeze(0)  # [S, 3, H, W]
        else:
            images = None

        num_frames = extrinsic.shape[0]
        print(f"  → Extracting {num_frames} frames")

        rgb_list = []
        depth_list = []
        pose_list = []
        intrinsics_list = []
        confidence_list = []
        sigma_list = []

        for i in range(num_frames):
            # RGB: [3, H, W] float [0,1] -> [H, W, 3] uint8
            if images is not None:
                rgb = images[i].transpose(1, 2, 0)
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                rgb_list.append(rgb)

            # Pose: 3x4 C2W -> 4x4 C2W
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :] = extrinsic[i].astype(np.float32)
            pose_list.append(pose)

            # Intrinsics: 3x3 K -> [fx, fy, cx, cy]
            K = intrinsic[i]
            intrinsics_list.append(np.array(
                [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32
            ))

            # Depth: [H, W, 1] -> [H, W]
            depth_frame = depth[i]
            if depth_frame.ndim == 3 and depth_frame.shape[-1] == 1:
                depth_frame = depth_frame.squeeze(-1)
            depth_list.append(depth_frame.astype(np.float32))

            # Confidence
            if "depth_conf" in predictions:
                conf = predictions["depth_conf"][0, i].float().cpu().numpy()
                confidence_list.append(conf.astype(np.float32))

            if "sigma_uv" in predictions:
                sigma = predictions["sigma_uv"][0, i].float().cpu().numpy()
                sigma_list.append(sigma.astype(np.float32))

        return rgb_list, depth_list, pose_list, intrinsics_list, confidence_list, sigma_list

    def __save_sigma_file__(self, output_dir: Path, base_name: str, data: np.ndarray) -> None:
        """Save per-frame sigma_uv maps as EXR."""
        output_dir.mkdir(parents=True, exist_ok=True)
        save_exr(np.asarray(data, dtype=np.float32), output_dir / f"{base_name}.exr")

    def process_scene(self, gt_artifact) -> Dict[str, Any]:
        """Process a scene with SURE-Map inference."""
        loader = BSSLoader(gt_artifact, resize_context=self.resize_context)
        input_rgb_list = loader.load_rgb_list()
        self.logger.info(f"Image size for processing: {loader.get_processing_dimensions()} (HxW)")

        print(f"  → Processing {len(input_rgb_list)} frames with SURE-Map (mode: {self.mode})")

        # Prepare and run inference
        images = self._prepare_images(input_rgb_list)
        image_shape = images.shape[-2:]  # (H, W)
        predictions = self._run_inference(images)

        # Convert outputs
        rgb_list, depth_list, pose_list, intrinsics_list, confidence_list, sigma_list = \
            self._process_outputs(predictions, image_shape)

        if len(depth_list) != len(input_rgb_list):
            print(f"  → WARNING: Output frames ({len(depth_list)}) != input frames ({len(input_rgb_list)})")

        # Assemble results
        print(f"  → Assembling {len(rgb_list)} frames in standard format")
        frame_results = {
            'rgb': rgb_list,
            'depth': depth_list,
            'pose': pose_list,
            'intrinsics': intrinsics_list,
        }

        if confidence_list:
            frame_results['confidence'] = confidence_list
        if sigma_list:
            frame_results['sigma'] = sigma_list

        return {
            'frame': frame_results,
            'global': {},
        }
