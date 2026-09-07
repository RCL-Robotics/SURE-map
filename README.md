# SURE-Map: Self-Correcting Streaming Geometric Foundation Models

SURE-Map equips streaming geometric foundation models with two complementary self-correction mechanisms:

- **Cross-view geometric uncertainty** assesses pose-depth correspondence consistency across consecutive views for dense-point filtering and local translation optimization.
- **Multi-timescale self-correction** couples uncertainty-weighted local translation optimization over consecutive frames with scale calibration from sparse keyframe-window inference, addressing both local pose errors and accumulated scale drift.

Together, they improve dense geometry and long-horizon trajectory accuracy while preserving streaming efficiency.

## Environment Setup

Please follow the environment setup instructions in [LingBot-Map](https://github.com/robbyant/lingbot-map).

## Model Checkpoints

The pretrained cross-view geometric uncertainty head is included in this repository and is available immediately after cloning:

```text
checkpoints/uncertainty.pt
```

The LingBot-Map backbone checkpoint is not included. Download the released `lingbot-map.pt` checkpoint following the official [LingBot-Map repository](https://github.com/robbyant/lingbot-map). SURE-Map uses this checkpoint as `checkpoints/lingbot.pt`:

```bash
wget -O checkpoints/lingbot.pt \
  https://huggingface.co/robbyant/lingbot-map/resolve/main/lingbot-map.pt
```

## Training

Download and extract the TartanAir v1 dataset from the official [TartanAir website](https://theairlab.org/tartanair-dataset/). The training command uses `--tartanair_root` to specify the extracted dataset directory.

### Cross-View Geometric Uncertainty

The backbone remains frozen. For simplicity, the paper uses backward-flow notation throughout. TartanAir provides forward flow, so training reverses the frame order relative to inference; the flow source frame always serves as the cross-attention query, aligning the uncertainty prediction with the supervised flow residual. We train the head on eight GPUs with 8-24-frame clips at `518 x 392`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  training/tests/train_flow_sigma_tartanair.py \
  --tartanair_root /path/to/tartanair_v1 \
  --ckpt checkpoints/lingbot.pt \
  --outdir training/outputs/flow_sigma_tartanair \
  --min_views 8 \
  --max_views 24 \
  --num_scale_frames 8 \
  --max_steps 20000 \
  --beta_nll 0.5
```

The resulting checkpoint is saved to:

```text
training/outputs/flow_sigma_tartanair/ckpts/tartanair_forward_step20000.pt
```

The included `checkpoints/uncertainty.pt` is the checkpoint used for the reported experiments. Additional checkpoints trained on broader datasets and more diverse temporal strides may be released in future updates.

## Evaluation Datasets

Follow the dataset preparation and evaluation protocol in [LingBot-Map](https://github.com/robbyant/lingbot-map) to download and prepare the following benchmarks:

- ✅ Oxford Spires dataset
- ✅ 7-Scenes dataset
- ✅ Neural RGB-D (NRGBD) dataset

For DTU evaluation, follow the [Spann3R data preprocessing guide](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md).

### KITTI Odometry

Download `data_odometry_color.zip`, `data_odometry_calib.zip`, and `data_odometry_poses.zip` from the official [KITTI Odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract all three archives into the same parent directory:

```bash
unzip data_odometry_color.zip -d /path/to/KITTI
unzip data_odometry_calib.zip -d /path/to/KITTI
unzip data_odometry_poses.zip -d /path/to/KITTI
```

The resulting dataset root is `/path/to/KITTI/dataset`, containing `sequences/` and `poses/`. Set `dataset.root` in `online/configs/kitti.yaml` to this directory.

### VBR

Download the processed VBR release used for long-horizon evaluation from [Junyi42/vbr_processed](https://huggingface.co/datasets/Junyi42/vbr_processed):

```bash
mkdir -p /path/to/vbr_parent
wget -c https://huggingface.co/datasets/Junyi42/vbr_processed/resolve/main/vbr_processed.tar.gz \
  -O /path/to/vbr_parent/vbr_processed.tar.gz
tar -xzf /path/to/vbr_parent/vbr_processed.tar.gz -C /path/to/vbr_parent
rm -f /path/to/vbr_parent/vbr_processed.tar.gz
```

The resulting dataset root is `/path/to/vbr_parent/vbr`, containing `{scene}_processed_aligned/` directories and `processed_gt/`. Set `raw_data_root` in `benchmark/configs/datasets/vbr.yaml` to this directory.

## Indoor Reconstruction

### Neural RGB-D

Set `raw_data_root` in `benchmark/configs/datasets/neural_rgbd.yaml` to the extracted Neural RGB-D directory. We use stride 5 and filter the 20% highest-uncertainty points.

```bash
cd benchmark

python prepare.py --config configs/sure_map_neural_rgbd.yaml
python run.py --config configs/sure_map_neural_rgbd.yaml
python eval_neural_rgbd_point_uncertainty.py \
  --config configs/sure_map_neural_rgbd.yaml
```

Results are saved to `benchmark/outputs/neural_rgbd/neural_rgbd_uncertainty_filter.json`.

### 7-Scenes

Set `raw_data_root` in `benchmark/configs/datasets/seven_scenes.yaml` to the extracted 7-Scenes directory. The default evaluation uses stride-5 streaming input. Since 7-Scenes contains relatively clean indoor scenes, we recommend filtering only the 10% highest-uncertainty points.

```bash
cd benchmark

python prepare.py --config configs/sure_map_seven_scenes.yaml
python run.py --config configs/sure_map_seven_scenes.yaml
python eval_seven_scenes_point_uncertainty.py \
  --config configs/sure_map_seven_scenes.yaml
```

Results are saved to `benchmark/outputs/seven_scenes/seven_scenes_uncertainty_filter.json`.

### DTU

Preprocess the DTU test set in MVSNet format following the [Spann3R guide](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md), then set `raw_data_root` in `benchmark/configs/datasets/dtu.yaml`. The default evaluation uses masked stride-1 input and filters the 20% highest-uncertainty points.

```bash
cd benchmark

python prepare.py --config configs/sure_map_dtu.yaml
python run.py --config configs/sure_map_dtu.yaml
python eval_dtu_point_uncertainty.py \
  --config configs/sure_map_dtu.yaml
```

Results are saved to `benchmark/outputs/dtu/dtu_uncertainty_filter.json`.

## Long-Horizon Pose Estimation

### Scene-Level Solver Configuration

We use a common solver formulation across datasets. Hessian validation is enabled for autonomous-driving datasets, where forward motion and near-planar geometry frequently cause local degeneracy. The scale-damping coefficient is configured at the dataset level, with the exact settings used in the paper provided in the corresponding YAML files.

### KITTI

Set `dataset.root` in `online/configs/kitti.yaml` to the KITTI odometry dataset directory, then run end-to-end streaming pose estimation and ATE evaluation with:

```bash
python online/run_kitti.py --config online/configs/kitti.yaml
```

Results are written to `online/outputs/kitti/`:

```text
kitti_<sequence>_sure_map_tum.txt
kitti_<sequence>_gt_tum.txt
summary.json
```

### Oxford Spires

Set `dataset.root` in `online/configs/oxford.yaml` to the Oxford Spires directory, then run:

```bash
python online/run_oxford.py --config online/configs/oxford.yaml
```

Results are written to `online/outputs/oxford/`:

```text
oxford_<sequence>_sure_map_tum.txt
oxford_<sequence>_gt_tum.txt
summary.json
```

### VBR Pose Estimation

Set `dataset.root` in `online/configs/vbr.yaml` to the processed VBR directory, then run:

```bash
python online/run_vbr.py --config online/configs/vbr.yaml
```

Results are written to `online/outputs/vbr/`:

```text
vbr_<sequence>_sure_map_tum.txt
vbr_<sequence>_gt_tum.txt
summary.json
```

## ✨ Acknowledgments

This work builds upon several excellent open-source projects:

- [VGGT](https://github.com/facebookresearch/vggt)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [LingBot-Map](https://github.com/robbyant/lingbot-map)
