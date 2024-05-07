_CONFIG = dict(
    NAME="fastvit_ma36",
    TAG="fastvit_ma36 deconv4 one conv k44",
    MODEL=dict(
        NAME="HandNet",
        BACKBONE=dict(
            model_name="fastvit_ma36",
            pretrain=True,
            out_feature_channel=1216,
            drop_path_rate=0.2,
        ),
        UV_HEAD=dict(
            in_features=1216, 
            out_features=42,
        ),
        DEPTH_HEAD=dict(
            in_features=1216, 
            out_features=1,
        ),
        MESH_HEAD=dict(
            in_channels=1216, 
            depths= [1, 1, 1],
            token_nums = [21, 84, 336],
            dims = [256, 128, 64],
            dropout=0.1,
            first_prenorms = [True, True, True],
            block_types = ["attention", "attention", "attention"], # attention, identity, conv
            # block_types = ["identity", "identity", "identity"], # attention, identity, conv
            
        ),
        LOSSES=dict(
            UV_LOSS_WEIGHT = 1.0,
            JOINTS_LOSS_WEIGHT = 10.0,
            DEPTH_LOSS_WEIGHT = 1.0,
            VERTICES_LOSS_WEIGHT = 10.0,            
        ),

    ),
    DATA=dict(
        IMAGE_SHAPE=[224, 224, 3],
        DATASET_DIR='/data/FreiHAND/training/rgb/',
        JSON_DIR='dataset/train.json',
        NORMALIZE_3D_GT=False,
        ROOT_INDEX=9,        
        AUG=dict(
            RandomChannelNoise=dict(noise_factor=0.4),
            RandomBrightnessContrastMap=dict(
                brightness_limit=(-0.4, 0.1),
                always_apply=True,
			),
            RandomHorizontalFlip=dict(flip_prob=0.5),
            BBoxCenterJitter=dict(
                factor=0.1, 
                dist='uniform'
            ),
            GetRandomScaleRotation=dict(
                rot_factor=90,
                min_scale_factor=1.0, 
                max_scale_factor=1.5, 
                rot_prob=1.0
            ),
        )
    ),
    TRAIN=dict(
        DATALOADER=dict(
            MINIBATCH_SIZE_PER_DIVICE=32,
            MINIBATCH_PER_EPOCH=256,
            NAME="train"
        ),
				
        WEIGHT_DECAY=1e-2,
        BASE_LR=5e-4,
        WARMUP_EPOCH=5,
        DUMP_EPOCH_INTERVAL=5,
        DO_EVAL=False,
        EVAL_EPOCH_INTERVAL=5,
        MAX_EPOCH=200,
        USE_CLEARML=True,
        CLIP_GRAD=dict(
           MAX_NORM=1,
        ),
        QAT=False,
        PRETRAIN="",
    ),
    VAL=dict(
        BMK=dict(
            name="FreiHand",
			json_dir='dataset/eval.json',
            eval_dir="/data/FreiHAND/evaluation/rgb/",
            scale_enlarge=1.25,
            ),
        IMAGE_SHAPE=(224, 224),
        BATCH_SIZE=64,
        ROOT_INDEX=0,
    ),
)