_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']


dataset_name = 'occ3d'
eval_metric = 'forecasting_miou'

class_weights = [0.0727, 0.0692, 0.0838, 0.0681, 0.0601, 0.0741, 0.0823, 0.0688, 0.0773, 0.0681, 0.0641, 0.0527, 0.0655, 0.0563, 0.0558, 0.0541, 0.0538, 0.0468] # occ-3d

occ_class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle','motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation','free']   # occ3d

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

grid_config = {
    'x': [-40, 40, 1.6],
    'y': [-40, 40, 1.6],
    'z': [-1, 5.4, 1.6],
    'depth': [1.0, 45.0, 0.5],
}

# In nuscenes data label is marked in 0.5s interval, so we can load future frame 6 to predict 3s future
train_load_future_frame_number = 6      # 0.5s interval 1 frame
train_load_previous_frame_number = 3    # 0.5s interval 1 frame
test_load_future_frame_number = 6       # 0.5s interval 1 frame
test_load_previous_frame_number = 3     # 0.5s interval 1 frame, 3 = 4 - 1

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# Running Config
num_gpus = 8
samples_per_gpu = 8 #16 #batch_size
workers_per_gpu = 20
total_epoch = 24 #48
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu)*4.554)      # total samples: 28130

# Model Config

# others params
num_classes = len(occ_class_names)
base_channel = 32
z_height = 16
class_embeds_dim = 16
n_e_ = 512
vq_frame_number = 4
previous_frame = 3
future_frame = 6
embed_dims = base_channel * 2
_ffn_dim_ = embed_dims * 2
pos_dim = embed_dims // 2

row_num_embed = 50  # latent_height
col_num_embed = 58  # latent_width

memory_frame_number = 9 # the final frame could condition on all 4 hist frames

task_mode = 'generate'
model = dict(
    type='II_World',
    previous_frame_exist=True if train_load_previous_frame_number > 0 else False,
    previous_frame=previous_frame,
    train_future_frame=train_load_future_frame_number,
    test_future_frame=test_load_future_frame_number,
    test_previous_frame=test_load_previous_frame_number,
    memory_frame_number=memory_frame_number,
    task_mode=task_mode,
    test_mode=False,
    class_weights = class_weights,
    feature_similarity_loss=dict(
        type='FeatSimLoss',
        loss_weight=1.0,
    ),
    recon_loss = dict(
        type='CustomFocalLoss',
        loss_weight=10.0,
    ),
    trajs_loss=dict(
        type='TrajLoss',
        loss_weight=0.01,
    ),
    rotation_loss=dict(
        type='RotationLoss',
        loss_weight=1.0,
    ),
    pose_encoder=dict(
        type='PoseEncoder',
        embed_dims= embed_dims,
        history_frame_number=memory_frame_number,
    ),
    transformer=dict(
        type='EE_Former',
        embed_dims=embed_dims,
        output_dims=embed_dims,
        use_gt_traj=True,
        use_transformation=True,
        history_frame_number=memory_frame_number,
        task_mode=task_mode,
        low_encoder=dict(
            type='EE_FormerEncoder',
            num_layers=5,
            transformerlayers=dict(
                type='EE_FormerEncoderLayer',
                use_plan=True,
                attn_cfgs=[
                    dict(
                        type='SelfAttention',
                        embed_dims=embed_dims,
                        dropout=0.0,
                        num_levels=1,
                        plan_aware = True,
                    ),
                    dict(
                        type='CrossPlanAttention',
                        embed_dims=embed_dims,
                        dropout=0.0,
                        num_levels=1,
                    )
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=embed_dims,
                    feedforward_channels=_ffn_dim_,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
            )
        ),
        high_encoder=dict(
            type='EE_FormerEncoder',
            num_layers=5,
            transformerlayers=dict(
                type='EE_FormerEncoderLayer',
                use_plan=False,
                attn_cfgs=[
                    dict(
                        type='SelfAttention',
                        embed_dims=embed_dims,
                        dropout=0.0,
                        num_levels=1,
                    ),
                    dict(
                        type='TemporalFusion',
                        embed_dims=embed_dims,
                        hisotry_number=memory_frame_number,
                        dropout=0.0,
                        num_levels=1,
                    )
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=embed_dims,
                    feedforward_channels=_ffn_dim_,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=('self_attn', 'norm', 'temporal_fusion', 'norm', 'ffn', 'norm')
            )
        ),
        positional_encoding=dict(
            type='PositionalEncoding',
            num_feats=pos_dim,
            row_num_embed=row_num_embed,
            col_num_embed=col_num_embed,
        )
    ),
    vqvae=dict(
        type='TriplaneVAE',
        occ_shape = (200,200,16),
        num_classes = num_classes,
        d_model = base_channel*2
    )
)

# Data
dataset_type = 'NuScenesWorldDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadStreamLatentToken', data_path='data/nuscenes/save_dir/token_vae'),
    dict(type='LoadStreamOcc3D'),
    dict(type='Collect3D', keys=['latent', 'voxel_semantics'])
]

test_pipeline = [
    dict(type='LoadStreamLatentToken', data_path='data/nuscenes/save_dir/token_vae'),
    dict(type='LoadStreamOcc3D'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent'])
]

share_data_config = dict(
    type=dataset_type,
    classes=occ_class_names,
    use_sequence_group_flag=True,
    # Eval Config
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    load_previous_data=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    load_future_frame_number=test_load_future_frame_number,
    load_previous_frame_number=test_load_previous_frame_number,
    ann_file=data_root + 'world-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'world-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=occ_class_names,
        test_mode=False,
        load_future_frame_number=train_load_future_frame_number,
        load_previous_frame_number=train_load_previous_frame_number,
        # Video Sequence
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
        # Set BEV Augmentation for the same sequence
        # bda_aug_conf=bda_aug_conf,
    ),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr =  1e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2),)

step_epoch = 20
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch*step_epoch,])

checkpoint_epoch_interval = 1
runner = dict(type='IterBasedRunner', max_iters=total_epoch * num_iters_per_epoch)
checkpoint_config = dict(interval=checkpoint_epoch_interval * num_iters_per_epoch)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=2*num_iters_per_epoch,
    ),
    dict(
        type='ScheduledSampling',
        total_iter=total_epoch * num_iters_per_epoch,
        loss_iter=None,
        # trans_iter=num_iters_per_epoch*step_epoch
    )
]

revise_keys = None