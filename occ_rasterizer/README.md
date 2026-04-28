# Occupancy Rasterizer Setup

## Data Processing

To render the ground-truth occupancy annotations in the validation split for evaluation, run:

```bash
torchrun --nproc_per_node=8 render_eval_condition_gt.py \
  --dataset_path data \
  --render_path data/gt_occ_conditions \
  --eval_split val
```

To render the ground-truth occupancy annotations in the training split for training, run:

```bash
torchrun --nproc_per_node=8 render_eval_condition_gt.py \
  --dataset_path data \
  --render_path data/gt_occ_conditions \
  --skip_existing \
  --eval_split train
```

To render the generated occupancy for video generation:

```bash
python render_eval_condition_gen.py --dataset_path [your nuscenes path] --generated_occ_path [your generated occupancy path] --eval_split val --vis
```

## Discription

This code is used to render the multi-view 2D sematic maps from the 3D occcupancy using Gaussian Splatting. The rendered 2D semantic maps are saved in the render_path.

#### --dataset_path

Path to the nuscenes dataset.

#### --render_path

Path to save the rendered 2D condition images.

#### --eval_split

Split to be rendered (train or val).

#### --vis

Visualize the rendered 2D condition images.

## Acknowledgement

Many thanks to these excellent projects:

- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [UniScene](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc?tab=readme-ov-file)

