_CONFIG = dict(
    NAME="convnext_base",
    MODEL=dict(
        NAME="HandNet",
        BACKBONE=dict(
            model_name="convnext_base",
            pretrain=True,
            out_feature_channel=1024,
        ),
        UV_HEAD=dict(
            in_channels=1024, 
            hidden_channels=256, 
            out_channels=42,            
            num_convs=0, 
            feat_shape=(7, 7),
            with_norm=True,
        ),
        DEPTH_HEAD=dict(
            in_channels=1024, 
            hidden_channels=256, 
            out_channels=1,       
            num_convs=0, 
            feat_shape=(7, 7),
            with_norm=True,
        ),
        JOINTS_HEAD=dict(
            in_channels=1024, 
            hidden_channels=256, 
            out_channels=63,            
            num_convs=0, 
            feat_shape=(7, 7),
            with_norm=True,
        ),
        LOSSES=dict(
            UV_LOSS_WEIGHT = 1.0,
            JOINTS_LOSS_WEIGHT = 10.0,
            DEPTH_LOSS_WEIGHT = 1.0,
        ),
    ),
    DATA=dict(
        IMAGE_SHAPE=[224, 224, 3],
        # (json_path, weight, data_type)
        # 0= aqy; 1=freihand; 2=interhand; 3=blender; 
        FILES=[            
            ("s3://fingerprint/handxr_data_v2/research_datas/jsons/single_hand_new/Freihand_train_joints.json", 1.0, 1), # FreiHand

            ("s3://fingerprint/handxr_data_v2/research_datas/jsons/single_hand_new/DexYcb_train_joints.json", 1.0, 1), # Dexycb
            
            ("s3://fingerprint/hands6dof-data/keypoint/jsons/single_hand/assembly_hands_train.json", 10.0, 2),

            ("s3://fingerprint/handxr_data_v2/research_datas/jsons/single_hand_new/InterhandSingle_train_joints.json", 1.0, 2), # IntagHand
            
            ("s3://fingerprint/handxr_data_v2/research_datas/jsons/single_hand_new/blender_base58_random_pose_all_joints.json", 5, 3), # Blender
            ("s3://fingerprint/handxr_data_v2/research_datas/jsons/single_hand_new/blender_freihand_all_joints.json", 5, 3), # Blender
            ("s3://fingerprint/handxr_data_v2/research_datas/jsons/single_hand_new/blender_interhand_all_joints.json", 5, 3), # Blender

        ],
        NORMALIZE_3D_GT=False,
        AUG=dict(
            RandomChannelNoise=dict(noise_factor=0.4),
            RandomBrightnessContrastMap=dict(
                aqy=(-0.1, 0.2),
                freihand=(-0.4, 0.1),
                interhand=(-0.4, 0.1),
                blender=(-0.1, 0.2),
            ),
            RandomHorizontalFlip=dict(flip_prob=0.5),
            BBoxCenterJitter=dict(
                factor=0.1, 
                dist='uniform'
            ),
            GetRandomScaleRotation=dict(
                rot_factor=90,
                min_scale_factor=1.0, 
                max_scale_factor=1.2, 
                rot_prob=1.0
            ),
        )
    ),
    TRAIN=dict(
        DPLINK=dict(
            MINIBATCH_SIZE_PER_DIVICE=64,
            MINIBATCH_PER_EPOCH=256,
            NAME="train"
        ),
        WEIGHT_DECAY=1e-2,
        BASE_LR=1e-3,
        WARMUP_EPOCH=10,
        DUMP_EPOCH_INTERVAL=5,
        DO_EVAL=False,
        EVAL_EPOCH_INTERVAL=5,
        MAX_EPOCH=300,
        USE_CLEARML=True,
        CLIP_GRAD=dict(
           MAX_NORM=1,
        ),
        QAT=False,
        PRETRAIN="",
    ),
    VAL=dict(
        BMKS=[
            dict(
                name="FreiHand",
                json_path = "s3://fingerprint/handxr_data_v2/research_datas/jsons/single_hand_new/Freihand_eval_joints.json",
                scale_enlarge=1.2,
            ),
            dict(
                name="dexycb",
                json_path = "s3://fingerprint/handxr_data_v2/research_datas/jsons/single_hand_new/DexYcb_test_joints.json",
                scale_enlarge=1.2,
            ), 
            dict(
                name="assembly",
                json_path = "s3://fingerprint/hands6dof-data/keypoint/jsons/single_hand/assembly_hands_val.json",
                scale_enlarge=1.2,
            ),             
        ],
        IMAGE_SHAPE=(224, 224),
        BATCH_SIZE=32,
    ),

)
