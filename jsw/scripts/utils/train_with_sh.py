# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/boostcampaitech7/level2-objectdetection-cv-04/blob/main/mmdetection3/train.py 
# 저거도 참고하면 좋을 듯.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register_all_modules(init_default_scope=True)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    

    



    # Runner 생성 전, train dataset build 전에
    # cfg.train_dataloader.dataset.metainfo['classes'] = [
    #     "General trash", "Paper", "Paper pack", "Metal", "Glass",
    #     "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
    # ]
    
    classes = [
    "General trash", "Paper", "Paper pack", "Metal", "Glass",
    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
    ]

    # 존재하지 않으면 metainfo 새로 생성
    ds = cfg.train_dataloader.dataset
    if not hasattr(ds, 'metainfo') or ds.metainfo is None:
        ds.metainfo = dict()

    ds.metainfo['classes'] = classes



    # Runner 생성 전, cfg merge 후
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    # dataset build
    from mmdet.registry import DATASETS

    train_ds_cfg = cfg.train_dataloader.dataset
    train_ds = DATASETS.build(train_ds_cfg)

    # dataset 크기 확인
    print(f"[DEBUG] Train dataset length: {len(train_ds)}")

    # batch_size 확인
    batch_size = cfg.train_dataloader.batch_size
    print(f"[DEBUG] train_dataloader.batch_size: {batch_size}")

    # drop_last, shuffle 확인
    drop_last = cfg.train_dataloader.get('drop_last', False)
    shuffle = cfg.train_dataloader.get('shuffle', True)
    print(f"[DEBUG] train_dataloader.drop_last: {drop_last}")
    print(f"[DEBUG] train_dataloader.shuffle: {shuffle}")

    # iteration 수 계산 (epoch 당)
    num_samples = len(train_ds)
    if drop_last:
        num_iters = num_samples // batch_size
    else:
        num_iters = (num_samples + batch_size - 1) // batch_size
    print(f"[DEBUG] Expected iterations per epoch: {num_iters}")

    # test dataset도 확인
    test_ds_cfg = cfg.test_dataloader.dataset
    test_ds = DATASETS.build(test_ds_cfg)
    print(f"[DEBUG] Test dataset length: {len(test_ds)}")
    print(f"[DEBUG] test_dataloader.batch_size: {cfg.test_dataloader.batch_size}")







    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # 251204 jsw
    # pipeline 수정
    cfg.train_dataloader.dataset.pipeline[2]['scale'] = (512, 512)
    cfg.test_dataloader.dataset.pipeline[1]['scale'] = (512, 512)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
