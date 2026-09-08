# SURE-Map: Self-Correcting Streaming Geometric Foundation Models

SURE-Map equips streaming geometric foundation models with two complementary self-correction mechanisms:

- **Cross-view geometric uncertainty** assesses pose-depth correspondence consistency across consecutive views for dense-point filtering and local translation optimization.
- **Multi-timescale self-correction** couples uncertainty-weighted local translation optimization over consecutive frames with scale recalibration from sparse keyframe-window inference, addressing both local pose errors and accumulated scale drift.

Together, they improve dense geometry and long-horizon trajectory accuracy while preserving streaming efficiency.

## Environment Setup

**1. Clone SURE-Map and create the conda environment**

```bash
git clone https://github.com/milchstrasse565/SURE-map.git SURE-Map
cd SURE-Map

conda create -n SURE-Map python=3.10 -y
conda activate SURE-Map
```

**2. Install PyTorch (CUDA 12.8)**

```bash
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions, select the corresponding installation command from [PyTorch](https://pytorch.org/get-started/locally/).

**3. Install SURE-Map**

```bash
pip install -e .
```

**4. Install FlashInfer**

FlashInfer provides paged KV-cache attention for efficient streaming inference:

```bash
pip install --index-url https://pypi.org/simple flashinfer-python
```

## Model Checkpoints

SURE-Map is designed for VGGT-style geometric foundation models. We provide the trained SURE-Map cross-view geometric uncertainty checkpoint used in the reported experiments:

```text
checkpoints/uncertainty.pt
```

To reproduce the experiments reported in this paper, download the backbone checkpoint:

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

The included `checkpoints/uncertainty.pt` is the checkpoint used for the reported experiments. Additional checkpoints trained on broader datasets and more diverse temporal strides will be released in future updates.

## Evaluation Datasets

Download the evaluation datasets from their official releases and prepare them as described below.

### Oxford Spires

Download the camera images, calibration files, and processed ground-truth trajectories from the [Oxford Spires dataset](https://ori-drs.github.io/datasets/oxford-spires/). Keep the following input layout, leaving each `images.zip` archive unextracted:

```text
oxford_spires_dataset/
  calibration/
    cam0.yaml
    cam-lidar-imu.yaml
  sequences/
    2024-03-12-keble-college-02/
      raw/images.zip
      processed/trajectory/gt-tum.txt
    ...
```

For trajectory evaluation, run the included preprocessing script from the repository root:

```bash
pip install open3d pyyaml
python preprocess/oxford.py \
  --dataset_dir /path/to/oxford_spires_dataset \
  --output_dir /path/to/oxford_spires \
  --max_frames 3840 \
  --images_only
```

This rectifies the cam0 images and exports matched camera-to-world ground-truth poses for evaluation. TLS maps and depth generation are not required. The output contains one directory per sequence (e.g., `keble-college-02/`), each with `images/`, `poses_c2w.txt`, and `intrinsics.txt`. Set `dataset.root` in `online/configs/oxford.yaml` to `/path/to/oxford_spires`.

### 7-Scenes and Neural RGB-D

- [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/): download and extract the original scene archives, including the nested `seq-XX.zip` archives. Keep the `scene/seq-XX/frame-*.color.png`, `frame-*.depth.png`, and `frame-*.pose.txt` files, along with each scene's `TestSplit.txt`.
- [Neural RGB-D (NRGBD)](https://github.com/dazinovic/neural-rgbd-surface-reconstruction#dataset): download and extract `neural_rgbd_data.zip`. Keep each scene's `images/`, `depth/`, `poses.txt`, and `focal.txt`.

Both loaders accept these original extracted formats. Set `raw_data_root` in the corresponding dataset YAML to the directory containing the scenes, then run the `prepare.py` commands under **Indoor Reconstruction** below.

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

Lower Acc./Comp./CD and higher F1 are better. F1 uses a 0.05 m distance threshold after point-cloud alignment.

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

**Paper results:** Tables III and V report the following Neural RGB-D results:

| Output variant | Paper variant | Acc. (m) | CD (m) | F1 (%) |
|---|---|---:|---:|---:|
| `raw` | SURE-Map w/o uncertainty filtering | 0.074 | 0.052 | 65.10 |
| `unc_filter` | SURE-Map | **0.067** | **0.051** | **66.20** |

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

**Paper results:** Tables III and V report the following 7-Scenes results:

| Output variant | Paper variant | Acc. (m) | CD (m) | F1 (%) |
|---|---|---:|---:|---:|
| `raw` | SURE-Map w/o uncertainty filtering | 0.035 | **0.039** | 81.77 |
| `unc_filter` | SURE-Map | **0.033** | **0.039** | **81.93** |

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

**Additional DTU results (not reported in the paper):**

| Method | Acc. (m) | Comp. (m) | F1 (%) |
|---|---:|---:|---:|
| LingBot-Map | 0.042 | **0.009** | 92.47 |
| SURE-Map | **0.015** | 0.011 | **94.80** |

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

**Paper results:** Table I reports per-sequence KITTI ATE-RMSE, and Table II reports the mean. The **SURE-Map** row (without LC) has a mean ATE-RMSE of **17.24 m** after Sim(3) alignment. Compare the per-sequence `ate_rmse_m` values and `mean_ate_rmse_m` in `summary.json` with these tables; lower is better.

### Oxford Spires

Set `dataset.root` in `online/configs/oxford.yaml` to the preprocessed Oxford Spires output directory described above, then run:

```bash
python online/run_oxford.py --config online/configs/oxford.yaml
```

Results are written to `online/outputs/oxford/`:

```text
oxford_<sequence>_sure_map_tum.txt
oxford_<sequence>_gt_tum.txt
summary.json
```

**Paper results:** Table II, **SURE-Map** row (without LC): mean ATE-RMSE **4.74 m** after Sim(3) alignment. The corresponding output is `mean_ate_rmse_m` in `summary.json`; lower is better.

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

**Paper results:** Table II, **SURE-Map** row (without LC): mean ATE-RMSE **28.58 m** after Sim(3) alignment. The corresponding output is `mean_ate_rmse_m` in `summary.json`; lower is better.

## License

Unless otherwise noted, the code in this repository is released under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.

Third-party components retain their original licenses. DUSt3R-derived files under `training/dust3r/` that carry CC BY-NC-SA 4.0 notices remain subject to that [license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## ✨ Acknowledgments

This work builds upon several excellent open-source projects:

- [VGGT](https://github.com/facebookresearch/vggt)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [LingBot-Map](https://github.com/robbyant/lingbot-map)
