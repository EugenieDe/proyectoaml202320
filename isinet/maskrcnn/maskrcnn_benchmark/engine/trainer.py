# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

import pdb


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, ((img_x, mask_x),
                    (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2, _),
                    (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _, _)) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        img_x, mask_x = img_x.to(device), mask_x.to(device)
        img_u_w = img_u_w.to(device)
        img_u_s1, img_u_s2 = img_u_s1.to(device), img_u_s2.to(device)
        cutmix_box1, cutmix_box2 = cutmix_box1.to(device), cutmix_box2.to(device)
        img_u_w_mix = img_u_w_mix.to(device) 
        img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.to(device), img_u_s2_mix.to(device)
        
        with torch.no_grad():
            model.eval()

            pred_u_w_mix = model(img_u_w_mix).detach()
            conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

        img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
            img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
        img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
            img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
        
        model.train()
        
        loss_dict_x = model(img_x, mask_x)
        pred_u_w = model(img_u_w, targets=None) 
        breakpoint()
        pred_u_w_fp = model(img_u_w, targets=None, need_fp=True)             

        pred_u_s1 = model(img_u_s1, ) 
        pred_u_s2 = model(img_u_s2, )

        pred_u_w = pred_u_w.detach()
        conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
        mask_u_w = pred_u_w.argmax(dim=1)

        mask_u_w_cutmixed1, conf_u_w_cutmixed1 = \
            mask_u_w.clone(), conf_u_w.clone()
        mask_u_w_cutmixed2, conf_u_w_cutmixed2 = \
            mask_u_w.clone(), conf_u_w.clone()

        mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
        conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

        mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
        conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

        #images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        loss_dict_s1 = model(pred_u_s1, mask_u_w_cutmixed1)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
