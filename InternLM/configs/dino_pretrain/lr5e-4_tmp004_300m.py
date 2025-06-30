JOB_NAME = "dino_lr5e-4_tmp004_300M"
N_GPUS = 16

PERFRAME_ATTEN = True
USE_CLS_TOKEN = True  # remember to change kd config, loss type, and dataset type
USE_VQ_EMBEDDING = False  # whether use pretrained VQGAN embedding
USE_VQ_ENCODER = False  # whether use pretrained VQGAN encoder
USE_CONV_STEM = True  # use convolutional stem as image encoder
USE_MULTI_CROP = True  # use multi-crop for DINO training
USE_TRACKING = False  # use the tracking mechanism introduced in DoRA
LEARNING_RATE = 5e-04
LR_ETA_MIN = 5e-06
TEMPERATURE = 0.04
BATCH_SIZE = 64
BATCH_ACCUM = 4
EPOCHS = 101
N_FRAMES = 1  # whether introduce temporal dimension, when N_FRAMES = 1, it treats each frame as a separate image sample
N_LOCAL_CROPS = 6  # number of local crops
IMAGE_SIZE = 224
N_FRAME_TOKENS = (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16)
N_LOCAL_TOKENS = (IMAGE_SIZE // 32) * (IMAGE_SIZE // 32)
# N_VIDEO_FRAMES = 395600
FPS = 30
# LEN_DATASET = N_VIDEO_FRAMES - (N_FRAMES * FPS)
LEN_DATASET = 7614851
ONE_EPOCH_ITERS = LEN_DATASET // (BATCH_SIZE * N_GPUS)
# remenber to change the total steps and ckpt steps when changing batch size
USE_FLASH_ATTN = True  # false for DoRA, since flash attention does not support its tracking mechanism
USE_CAUSAL_ATTN = True
FRAMES_2_BATCH = False

DO_ALERT = False
VQGAN_FOLDER = 'resources/vqgan_model'

########################################################
#           Teacher Model Config
########################################################

teacher_type = "INTERNLM_VQGAN"
teacher_ckpt_folder = f"local:llm_teacher_ckpts/{JOB_NAME}"
kd_config = dict(gt_weight=0., kd_weight=1., type='dinoimgcls', temperature=0.996)

T_HIDDEN_SIZE = 1024
T_OUTPUT_DIM = 65536
T_NUM_ATTENTION_HEAD = 8
T_MLP_RATIO = 8/3
T_NUM_LAYER = 22
T_VOCAB_SIZE = 8192 # 8192 (img) + 256 (pose) + 1 ([img]) + 1 ([pose]) + 10 (extra)

teacher = dict(
    teacher=True,
    perframe_atten=PERFRAME_ATTEN,
    use_cls_token=USE_CLS_TOKEN,
    use_multi_crop=USE_MULTI_CROP,
    use_tracking=USE_TRACKING,
    head_type="dino",  # swiglu, dino
    checkpoint=False,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
    num_attention_heads=T_NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=T_VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=T_HIDDEN_SIZE,
    output_dim=T_OUTPUT_DIM,
    num_layers=T_NUM_LAYER,
    mlp_ratio=T_MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.float16",  # Support: "torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.tf32"
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-6,
    use_flash_attn=USE_FLASH_ATTN,
    return_track=not USE_FLASH_ATTN,
    causal=USE_CAUSAL_ATTN,
    frames2batch=FRAMES_2_BATCH,
    num_chunks=1,  # if num_chunks > 1, interleaved pipeline scheduler is used.
    lvm_config=dict(
        enable=USE_VQ_EMBEDDING,
        embedding_cfg=dict(
            vq_model_path=VQGAN_FOLDER,
            embedding_dim=T_HIDDEN_SIZE,
            freeze_vq_model=True,
        ),
        enable_conv_stem=USE_CONV_STEM,
        conv_stem_config=dict(
            img_size=IMAGE_SIZE,
            patch_size=16,
        ),
        enable_encoder=USE_VQ_ENCODER,
        encoder_cfg=dict(
            vq_model_path=VQGAN_FOLDER,
            freeze_vq_model=True,
        ),
    )
)

momentum_scheduler = dict(
    momentum_teacher=0.996,
    epochs=EPOCHS,
)

########################################################
#           Student Model Config
########################################################
model_type = "INTERNLM_VQGAN"

SEQ_LEN = 1
HIDDEN_SIZE = 1024
OUTPUT_DIM = 65536
NUM_ATTENTION_HEAD = 8
MLP_RATIO = 8/3
NUM_LAYER = 22
VOCAB_SIZE = 8192 # 8192 (img) + 256 (pose) + 1 ([img]) + 1 ([pose]) + 10 (extra)

model = dict(
    teacher=False,
    perframe_atten=PERFRAME_ATTEN,
    use_cls_token=USE_CLS_TOKEN,
    use_multi_crop=USE_MULTI_CROP,
    use_tracking=USE_TRACKING,
    head_type="dino",  # swiglu, dino
    checkpoint=1,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    output_dim=OUTPUT_DIM,
    num_layers=NUM_LAYER,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.float16",  # Support: "torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.tf32"
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-6,
    use_flash_attn=USE_FLASH_ATTN,
    return_track=not USE_FLASH_ATTN,
    causal=USE_CAUSAL_ATTN,
    frames2batch=FRAMES_2_BATCH,
    num_chunks=1,  # if num_chunks > 1, interleaved pipeline scheduler is used.
    lvm_config=dict(
        enable=USE_VQ_EMBEDDING,
        embedding_cfg=dict(
            vq_model_path=VQGAN_FOLDER,
            embedding_dim=HIDDEN_SIZE,
            freeze_vq_model=True,
        ),
        enable_conv_stem=USE_CONV_STEM,
        conv_stem_config=dict(
            img_size=IMAGE_SIZE,
            patch_size=16,
        ),
        enable_encoder=USE_VQ_ENCODER,
        encoder_cfg=dict(
            vq_model_path=VQGAN_FOLDER,
            freeze_vq_model=True,
        ),
    )
)

########################################################
#           Checkpopint and Dataset Config
########################################################

MODEL_ONLY_FOLDER = f"local:llm_ckpts/{JOB_NAME}"
SAVE_CKPT_FOLDER = f"local:save_ckpt/{JOB_NAME}"
SAVE_TEACHER_CKPT_FOLDER = f"local:save_ckpt/{JOB_NAME}_teacher"
LOAD_CKPT_FOLDER = f"local:save_ckpt/{JOB_NAME}"

CHECKPOINT_EVERY = ONE_EPOCH_ITERS

ckpt = dict(
    enable_save_ckpt=True,  # set True to enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    save_teacher_ckpt_folder=SAVE_TEACHER_CKPT_FOLDER,
    # load_ckpt_folder= dict(path=MODEL_ONLY_FOLDER, content=["all"], ckpt_type="normal"),
    # load_ckpt_folder="local:llm_ckpts/",
    # 'load_ckpt_info' setting guide:
    # 1. the 'path' indicate ckpt path,
    # 2. the 'content‘ means what states will be loaded, support: "model", "sampler", "optimizer", "scheduler", "all"
    # 3. the ’ckpt_type‘ means the type of checkpoint to be loaded, now only 'normal' type is supported.
    # load_ckpt_info=dict(path=MODEL_ONLY_FOLDER, content=("model",), ckpt_type="internlm"),
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    # oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
    oss_snapshot_freq=0,
)

TRAIN_FOLDER = [
    "data/WTour/Walking_Tour_Venice.mp4",
    "data/WTour/Walking_Tour_Amsterdam.mp4",
    "data/WTour/Walking_Tour_Bangkok.mp4",
    "data/WTour/Walking_Tour_Chiang_Mai.mp4",
    "data/WTour/Walking_Tour_Istanbul.mp4",
    "data/WTour/Walking_Tour_Kuala_Lumpur.mp4",
    "data/WTour/Walking_Tour_Singapore.mp4",
    "data/WTour/Walking_Tour_Stockholm.mp4",
    "data/WTour/Walking_Tour_Wildlife.mp4",
    "data/WTour/Walking_Tour_Zurich.mp4"
    ]
VALID_FOLDER = []

data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=BATCH_SIZE,
    # packed_length = micro_bsz * SEQ_LEN
    micro_bsz=N_FRAMES,
    # defaults to the value of micro_num
    valid_micro_num=2,
    # defaults to 0, means disable evaluate
    valid_every=0,
    gradient_accumulation=BATCH_ACCUM,
    pack_sample_into_one=False,
    train_one_epoch=False,
    total_steps=ONE_EPOCH_ITERS*EPOCHS,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    train_folder=TRAIN_FOLDER,
    valid_folder=None,
    empty_cache_and_diag_interval=10000,
    diag_outlier_ratio=1.1,
    padding_token=8331,
    frames_per_clip=N_FRAMES,
    dataset_type="WTVideos",  # WT1Video, WTVideos, ImgNet, ImgNetDINO
    step_between_clips=FPS,
    global_crops_scale=(0.4, 1.),
    use_local_crop=USE_MULTI_CROP,
    use_egoexo=True,
    local_crops_scale=(0.05, 0.4),
    local_crops_number=N_LOCAL_CROPS,
    # first crop an image with initial size
    init_crop_size=300,
    # then crop global views from the image
    vid_crop_size=IMAGE_SIZE,
    # finally crop local views from the image
    local_crop_size=IMAGE_SIZE // 2,
    norm_vid=True,
    num_frame_tokens=N_FRAME_TOKENS,
)

########################################################
#           Optimizer Config
########################################################
grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=False,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

# DINO loss configuration
loss = dict(
    name='dinoimgcls',
    incld_cls_token=USE_CLS_TOKEN,
    use_multicrop=USE_MULTI_CROP,
    use_tracking=USE_TRACKING,
    out_dim=65536,
    local_crops_number=N_LOCAL_CROPS,
    warmup_teacher_temp=TEMPERATURE,
    teacher_temp=TEMPERATURE,
    warmup_teacher_temp_epochs=3,
    epochs=EPOCHS,
    grad_accum=BATCH_ACCUM,
    batch_size=BATCH_SIZE//BATCH_ACCUM,
    global_token_len=N_FRAME_TOKENS,
    local_token_len=N_LOCAL_TOKENS,
    nframes=N_FRAMES,
)

adam = dict(
    lr=LEARNING_RATE,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.04,
    weight_decay_end=0.4,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.1,
    eta_min=LR_ETA_MIN,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

"""
zero1 parallel:
    1. if zero1 <= 0, The size of the zero process group is equal to the size of the dp process group,
        so parameters will be divided within the range of dp.
    2. if zero1 == 1, zero is not used, and all dp groups retain the full amount of model parameters.
    3. zero1 > 1 and zero1 <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
pipeline parallel (dict):
    1. size: int, the size of pipeline parallel.
    2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler.
tensor parallel: tensor parallel size, usually the number of GPUs per node.
"""
parallel = dict(
    zero1=1,  # 1 for debug on single GPU, default is 8
    pipeline=dict(size=1, interleaved_overlap=True),
    sequence_parallel=False,
)

cudnn_deterministic = False
cudnn_benchmark = False

monitor = dict(
    # feishu alert configs
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
    ),
)