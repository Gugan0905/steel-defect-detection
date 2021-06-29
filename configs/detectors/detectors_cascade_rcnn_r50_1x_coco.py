'''_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')))'''

base = [
    'base/models/cascade_rcnn_r50_fpn.py',
    #'../base/datasets/coco_detection.py',
    #'../base/schedules/schedule_1x.py', '../base/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='DetectoRS_ResNeXt',
        #conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            #conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True),
           
            pretrained='open-mmlab://resnext101_32x4d',
            style='pytorch')))

dataset_type = 'CustomDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        multiscale_mode='range',
        img_scale=[(1333, 800), (1333, 1200)],
        keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0)
                ],
                p=0.1),
            dict(
                type='JpegCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1333, 1200)],
        transforms=[
            dict(type='Resize', multiscale_mode='range', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=1.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CustomDataset',
        ann_file='data/train_80_mmdetection.pkl',
        img_prefix='data/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='Resize',
                multiscale_mode='range',
                img_scale=[(1333, 800), (1666, 1000)],
                keep_ratio=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0625,
                        scale_limit=0.0,
                        rotate_limit=0,
                        interpolation=1,
                        p=0.5),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                r_shift_limit=10,
                                g_shift_limit=10,
                                b_shift_limit=10,
                                p=1.0),
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='JpegCompression',
                        quality_lower=85,
                        quality_upper=95,
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.1)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor'))
        ]),
    val=dict(
        type='CustomDataset',
        ann_file='data/val_10_mmdetection.pkl',
        img_prefix='data/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1333, 800), (1666, 1000)],
                transforms=[
                    dict(
                        type='Resize',
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=1.0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        ann_file='data/test_10_mmdetection.pkl',
        img_prefix='data/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1333, 200), (1999, 1000)],
                transforms=[
                    dict(
                        type='Resize',
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=1.0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
checkpoint_config = dict(interval=1)
log_config = dict(interval=64, hooks=[dict(type='TextLoggerHook')])
total_epochs = 30
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_rcnn_x101_32x4d_fpn_1x_detectoRS_RFP'
load_from = None
resume_from = None
workflow = [('train', 1)]
