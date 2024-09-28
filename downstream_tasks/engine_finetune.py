# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
from typing import Iterable
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import torch
import torch.nn.functional as F

from timm.utils import accuracy

import downstream_tasks.util.misc as misc
import downstream_tasks.util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        context_autocast = torch.cuda.amp.autocast()
            
        with context_autocast: 
            outputs = model(samples).squeeze()
            if args.task == 'classification':
                if len(outputs.shape) == 1:
                    outputs = outputs.unsqueeze(0)
            try:
                if len(targets) == 1:
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets.squeeze())
                    targets = targets.squeeze()
            except UserWarning as e:
                print(f"Caught an exception: {e}")
                raise Exception('error')
        loss_value = loss.item()
        
        if args.task == 'regression':
            mse = torch.mean((targets - outputs.squeeze()) ** 2)
            r2 = torch.sum((targets - torch.mean(targets, dim=0)) * (outputs.squeeze() - torch.mean(outputs.squeeze(), dim=0))) / (torch.sqrt(torch.sum((targets - torch.mean(targets, dim=0)) ** 2)) * torch.sqrt(torch.sum((outputs.squeeze() - torch.mean(outputs.squeeze(), dim=0)) ** 2)))
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if args.task == 'regression':
            metric_logger.update(r2=r2)
            metric_logger.update(mse=mse)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, task):
    if task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    gt_all = []
    predict_class_all = []
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        context_autocast = torch.cuda.amp.autocast()
        
        with context_autocast: 
            output = model(images).squeeze()
            if task == 'classification':
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
            try:
                if len(target) == 1:
                    loss = criterion(output, target)
                else:
                    loss = criterion(output, target.squeeze())
            except UserWarning as e:
                print(f"Caught an exception: {e}")
                raise Exception('error')
            

        if task == 'classification':
            acc1 = accuracy(output, target, topk=(1,))[0]
            probabilities = F.softmax(output, dim=1)
            gt = target.detach().cpu().numpy()
            predict = probabilities.detach().cpu().numpy()

            gt_all.append(gt)
            predict_class_all.append(predict)

        else:
            gt = target.detach().cpu().numpy()
            predict = output.detach().cpu().numpy()
            gt_all.append(gt)
            predict_class_all.append(predict)
            
            mse = torch.mean((target - output.squeeze()) ** 2)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        
        if task == 'classification':
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            
        else:
            metric_logger.update(mse=mse.item())

            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    if task == 'classification':
        gt = np.concatenate(gt_all)
        predict_class = np.concatenate(predict_class_all, axis=0)
        predict = np.argmax(predict_class, axis=1)
        
        f1 = f1_score(gt, predict)
        metric_logger.update(f1=f1.item())
        
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} F1 {f1.global_avg:.3f}'
            .format(top1=metric_logger.acc1, losses=metric_logger.loss, f1=metric_logger.f1))
    else:
        target = torch.from_numpy(np.concatenate(gt_all))
        output = torch.from_numpy(np.concatenate(predict_class_all, axis=0)).float()
        r2 = torch.sum((target - torch.mean(target, dim=0)) * (output.squeeze() - torch.mean(output.squeeze(), dim=0))) / (torch.sqrt(torch.sum((target - torch.mean(target, dim=0)) ** 2)) * torch.sqrt(torch.sum((output.squeeze() - torch.mean(output.squeeze(), dim=0)) ** 2)))
        metric_logger.update(r2=r2.item())
        print('* MSE {losses.global_avg:.4f} R2 {r2.global_avg:.4f}'
            .format(losses=metric_logger.loss, r2=metric_logger.r2))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
