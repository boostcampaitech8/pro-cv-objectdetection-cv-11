python train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
  --work-dir ./work_dirs/faster_rcnn_r50_fpn_1x_trash \
  --cfg-options \
    train_dataloader.batch_size=4 \
    train_dataloader.num_workers=2 \
    train_dataloader.dataset.data_root="../../../dataset/" \
    train_dataloader.dataset.ann_file="train.json" \
    train_dataloader.dataset.data_prefix.img="" \
    val_dataloader=None \
    val_evaluator=None \
    val_cfg=None \
    val_loop=None \
    test_dataloader.batch_size=1 \
    test_dataloader.num_workers=2 \
    test_dataloader.dataset.data_root="../../../dataset/" \
    test_dataloader.dataset.ann_file="test.json" \
    test_dataloader.dataset.data_prefix.img="" \
    model.roi_head.bbox_head.num_classes=10 \
    train_cfg.max_epochs=12 \
    optim_wrapper.clip_grad.max_norm=35 \
    optim_wrapper.clip_grad.norm_type=2 \
    default_hooks.checkpoint.max_keep_ckpts=3 \
    default_hooks.checkpoint.interval=1 \
    gpu_ids="[0]" \
    device="cuda"

# 에러 1)
# train_dataloader.dataset.metainfo.classes="['General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']" \
# 저거 중첩 list라서 적용 안되고 자꾸 데이터셋 크기가 작게 잡힘
# 해결방법) 인자로 주지 말고 train.py에서 metainfo를 직접 수정

# 에러2) 저거 저렇게 하니까 에러 남.
# train_dataloader.dataset.pipeline[2].scale=[512,512] \
# test_dataloader.dataset.pipeline[1].scale=[512,512] \
# 에러원인)
# train_dataloader.dataset = {
# "pipeline[2]": {"scale": (512, 512)}
# } 로 인식돼서 생기는 에러
# 해결방법) 결국 train.py 내부에서 직접 바꿔줘야 함.

# 결론) 인자를 줄 때 [ ] 기호가 들어가는 경우는 문자열로 다 환산돼버리니까 인자로 주지 말고 직접 python 파일 수정하자.

# python train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#   --work-dir ./work_dirs/faster_rcnn_r50_fpn_1x_trash \
#   --cfg-options \
#   train_dataloader.dataset.metainfo.classes="['General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']" \
#   train_dataloader.batch_size=4 \
#   train_dataloader.num_workers=2 \
#   train_dataloader.dataset.data_root="../../../dataset/" \
#   train_dataloader.dataset.ann_file="train.json" \
#   train_dataloader.dataset.data_prefix.img="" \
#   val_dataloader=None \
#   val_evaluator=None \
#   val_cfg=None \
#   val_loop=None \
#   test_dataloader.batch_size=1 \
#   test_dataloader.num_workers=2 \
#   test_dataloader.dataset.data_root="../../../dataset/" \
#   test_dataloader.dataset.ann_file="test.json" \
#   test_dataloader.dataset.data_prefix.img="" \
#   model.roi_head.bbox_head.num_classes=10 \
#   train_cfg.max_epochs=12 \
#   default_hooks.checkpoint.max_keep_ckpts=3 \
#   default_hooks.checkpoint.interval=1 \
#   gpu_ids="[0]" \
#   device="cuda"