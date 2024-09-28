# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import sys
sys.path.append('/home/users/nus/li.rl/scratch/code/ijepa')
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import downstream_tasks.util.misc as misc
from downstream_tasks.util.misc import NativeScalerWithGradNormCount as NativeScaler
from downstream_tasks.util.lars import LARS

from src.datasets.hca_sex_datasets import make_hca_sex

from downstream_tasks.models_vit import VisionTransformer

from downstream_tasks.engine_finetune import train_one_epoch, evaluate


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    if args.data_make_fn == 'hca_sex':
        if args.data_make_fn == 'hca_sex':
            data_fn = make_hca_sex
        else:
            raise "data function {} not implemented!"
        
        data_loader_train, data_loader_val, data_loader_test, train_dataset, valid_dataset, test_dataset = data_fn(
            batch_size=args.batch_size,
            pin_mem=args.pin_mem,
            num_workers=args.num_workers,
            world_size=1,
            rank=0,
            drop_last=False,
            data_split=[0.6, 0.2, 0.2],
            processed_dir=f'path/to/data',
            use_normalization=args.use_normalization,
            label_normalization=args.label_normalization,
            downsample=args.downsample
        )
        
    else:
        raise Exception('data make fn error')
    
    print(f'task: {args.data_make_fn}')
    print(f'len train dataset: {len(train_dataset)}')
    print(f'len validation dataset: {len(valid_dataset)}')
    print(f'len test dataset: {len(test_dataset)}')
    model = VisionTransformer(
        args,
        model_name=args.model_name,
        attn_mode=args.attn_mode,
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        device=device,
        add_w=args.add_w
    )

    if args.finetune and not args.eval:
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(args.finetune + "\n")
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['target_encoder']
        state_dict = model.state_dict()
        
        new_checkpoint_model = {}
        for key in checkpoint_model.keys():
            new_key = key.replace('module.', 'encoder.')  # Remove 'module.' from each key
            new_checkpoint_model[new_key] = checkpoint_model[key]

        for k in ['head.weight', 'head.bias']:
            if k in new_checkpoint_model and new_checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del new_checkpoint_model[k]

        # load pre-trained model
        msg = model.load_state_dict(new_checkpoint_model, strict=False)
        print(msg)
        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.6f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(args, data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(valid_dataset)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        val_stats = evaluate(args, data_loader_val, model, device, args.task)
        if args.task == 'classification':
            print(f"Accuracy of the network on the {len(valid_dataset)} validation samples: {val_stats['acc1']:.1f}%")
        else:
            print(f"MSE of the network on the {len(valid_dataset)} validation samples: {val_stats['loss']:.3f}, R2: {val_stats['r2']:.3f}")
            
        test_stats = evaluate(args, data_loader_test, model, device, args.task)
        if args.task == 'classification':
            print(f"Accuracy of the network on the {len(test_dataset)} test samples: {test_stats['acc1']:.1f}%")
        else:
            print(f"MSE of the network on the {len(test_dataset)} test samples: {test_stats['loss']:.3f}, R2: {test_stats['r2']:.3f}")
        
        if log_writer is not None:
            if args.task == 'classification':
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            else:
                log_writer.add_scalar('perf/test_mse', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

