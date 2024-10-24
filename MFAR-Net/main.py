import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from utils import misc
from utils.ema import ModelEMA
from utils.distributed_utils import select_device, init_distributed_mode, get_rank, is_main_process
from utils.tiny_utils import draw, draw_compare, save_evaluations, save_parameters
from data.dataset_builder import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models.MFAR_Net import build_model
from data.collate_fn import default_collate_fn
from models.ha_loss_generator import HALossGenerator


def get_args_parser():
    parser = argparse.ArgumentParser('Setting', add_help=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--output_dir', default='./runs',  # <------------------------save_path
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='0',
                        help='device to use for training / testing')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--checkpoint_step', default=1, type=int,
                        help="")
    parser.add_argument('--ema', default=True, type=bool,
                        help="")
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters--------------------------------------------------------------------------------------------------
    parser.add_argument('--num_classes', type=int, default=20,  # <------------------------num_classes
                        help="Number of object classes")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--return_idx', default=[1, 2, 3], type=list,
                        help="")
    parser.add_argument('--freeze_convs', default=True, type=bool,
                        help="")
    parser.add_argument('--freeze_norm', default=True, type=bool,
                        help="")
    parser.add_argument('--unfreeze_at', default=100, type=int,
                        help="")
    parser.add_argument('--pretrained', default="./weights/ResNet50.pth", type=str,
                        help="")
    # * Transformer
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0., type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")

    # Loss--------------------------------------------------------------------------------------------------------------
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--eos_coef', default=1e-4, type=float,
                        help="")
    parser.add_argument('--iou_type', default="SIoU", type=str,
                        help="")
    parser.add_argument('--with_cam_loss', action='store_true',
                        help="")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_iou', default=2, type=float,
                        help="iou box coefficient in the matching cost")
    parser.add_argument('--use_focal_loss', default=True, type=bool,
                        help="use focal loss")
    # * Loss coefficients
    parser.add_argument('--vfl_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--iou_loss_coef', default=2, type=float)

    # dataset parameters------------------------------------------------------------------------------------------------
    parser.add_argument('--coco_path', default="../datasets/NWPU VHR-10/nwpu", type=str)  # <------------------dataset
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--imgsize', default=[640, 640])
    parser.add_argument('--multi_scale', default=None)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters-----------------------------------------------------------------------------------
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    return parser


def main(args):
    if is_main_process():
        print(args)
    output_dir = Path(args.output_dir)
    save_parameters(args, output_dir)
    device = torch.device(select_device(args.device, args.batch_size))
    # print(device)
    init_distributed_mode(args)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.distributed:
        device = args.gpu
    model, criterion, postprocessors = build_model(args, device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process():
            print('Using SyncBatchNorm')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone, },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=default_collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=sampler_val,
                                 collate_fn=default_collate_fn, drop_last=False, num_workers=args.num_workers)
    if args.eval:
        dataset_test = build_dataset(image_set='test', args=args)
        if args.distributed:
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, sampler=sampler_test,
                                      collate_fn=default_collate_fn, drop_last=False, num_workers=args.num_workers)
        base_ds = get_coco_api_from_dataset(data_loader_test.dataset)
    else:
        base_ds = get_coco_api_from_dataset(data_loader_val.dataset)
    with open(os.path.join(output_dir, "val_transformed.json"), 'w') as f:
        json.dump(base_ds.dataset, f, indent=4)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['ema']["module"] if args.ema else checkpoint['model'])

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        checkpoint = checkpoint['ema']["module"] if args.ema else checkpoint['model']
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in model_without_ddp.state_dict().items():
            if k in checkpoint:
                if v.shape == checkpoint[k].shape:
                    matched_state[k] = checkpoint[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        model_without_ddp.load_state_dict(matched_state, strict=False)
        if is_main_process():
            print(f'Load model.state_dict, missed: {missed_list}, unmatched: {unmatched_list}')

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    if args.ema:
        ema = ModelEMA(model, decay=0.9999, warmups=2000)

    if args.eval:
        test_stats, coco_evaluator, _ = evaluate(model, criterion, postprocessors, data_loader_test, base_ds, device,
                                                 args.output_dir, args.num_classes)
        if args.output_dir:
            misc.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    if is_main_process():
        print("Start training")
    start_time = time.time()
    train_log = {"lr": [], "loss": [], "loss_vfl": [], "loss_bbox": [], "loss_iou": []}
    val_log = {"P_0.5-0.95": [], "P_0.5": [], "P_0.75": [], "P_S": [], "P_M": [], "P_L": [],
               "R_m1": [], "R_m10": [], "R_m100": [], "R_S": [], "R_M": [], "R_L": []}
    val_losses = {"loss_vfl": [], "loss_bbox": [], "loss_iou": [], }
    best_stat = {'epoch': -1, }
    if args.with_cam_loss:
        ha_loss_generator = HALossGenerator(model, [model.encoder.pan_blocks[-1],
                                                    model.encoder.pan_blocks[-2],
                                                    model.encoder.fpn_blocks[0]],
                                            "vit_b", "./weights/sam_vit_b.pth", device)
    else:
        ha_loss_generator = None

    for epoch in range(args.start_epoch, args.epochs):
        if epoch == args.unfreeze_at:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print(f"unfreeze backbone at[epoch:{epoch}]")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, ha_loss_generator, criterion, data_loader_train, optimizer, output_dir,
                                      device, epoch, args.clip_max_norm, ema=ema, with_cam_loss=args.with_cam_loss)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.checkpoint_step == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                misc.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        module = ema.module if args.ema else model

        test_stats, coco_evaluator, eval_losses = evaluate(module, criterion, postprocessors, data_loader_val, base_ds,
                                                           device, args.output_dir, args.num_classes)

        for k in test_stats.keys():
            if k in best_stat:
                best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                best_stat[k] = max(best_stat[k], test_stats[k][0])
            else:
                best_stat['epoch'] = epoch
                best_stat[k] = test_stats[k][0]
        if is_main_process():
            print('best_stat: ', best_stat)

        for k, v in train_stats.items():
            if k in train_log.keys():
                train_log[k].append(float(v))
        for k, v in test_stats.items():
            if k == "coco_eval_bbox":
                val_res = v
                for i, (log_k, log_v) in enumerate(val_log.items()):
                    val_log[log_k].append(float(val_res[i]))
        for k, v in eval_losses.items():
            if k in val_losses.keys():
                val_losses[k].append(float(v))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

            draw(train_log, output_dir, args, "train")
            draw(val_log, output_dir, args)
            draw(val_losses, output_dir, args, "eval")
            draw_compare(train_log, val_losses, output_dir, args, "compare")
            save_evaluations(val_log, output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if is_main_process():
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
