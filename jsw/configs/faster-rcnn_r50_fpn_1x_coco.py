_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# faster-rcnn_r50_fpn_1x_coco.py 안에 이미 모델 설정이 포함되어 있음.