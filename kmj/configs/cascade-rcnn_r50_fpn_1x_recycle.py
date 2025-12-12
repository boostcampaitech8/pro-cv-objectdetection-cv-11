_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

### custom ###
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MinIoURandomCrop',
        min_ious=[0.1, 0.3, 0.5, 0.7],
        min_crop_size=0.3
    ),
    dict(type='RandomShift', shift_ratio=0.2),
    dict(type='PackDetInputs')
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='cv_11_OD',
             entity='cv_11',
             name='faster_rcnn_test'
        )
    )
]

visualizer = dict(type='DetLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

