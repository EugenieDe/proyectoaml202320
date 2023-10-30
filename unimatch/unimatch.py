import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.nn.functional as F

from dataset.semi import SemiDataset
from dataset.endovis import EndovisDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from more_scenarios.medical.util.utils import DiceLoss

from more_scenarios.medical.model.unet import UNet

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--device', default='cuda:3')
#parser.add_argument('--port', default=None, type=int)

def visualize(image, pred, mask, name):
    image_c = image.detach().cpu().numpy()
    pred_c = pred.detach().cpu().numpy()
    mask_c = mask.detach().cpu().numpy()
    #breakpoint()
    pred_c = pred_c.argmax(1)
    image_c=image_c[0].transpose(1,2,0)
    pred_c=pred_c[0]
    mask_c = mask_c[0]
    fig, ax = plt.subplots(1,3)
    ax[0].set_title('Image',fontsize = 5)
    ax[0].imshow(image_c)
    ax[0].axis('off')

    ax[1].set_title('Ground Truth',fontsize = 5)
    ax[1].imshow(np.zeros(mask_c.shape),cmap='gray')
    #mask[mask==0]=np.nan
    ax[1].imshow(mask_c,cmap='Set1',norm=Normalize(1,10))
    ax[1].axis('off')

    ax[2].set_title('Pred',fontsize = 5)
    ax[2].imshow(np.zeros(pred_c.shape),cmap='gray')
    #pre[pre==0]=np.nan
    ax[2].imshow(pred_c,cmap='Set1',norm=Normalize(1,10))
    ax[2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig('train_image/{}.png'.format(name), dpi=400, bbox_inches='tight')
    plt.close()

def main():
    args = parser.parse_args()
    device = args.device
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    #rank, world_size = setup_distributed(port=args.port)
    rank, world_size = 0, 1

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    #cudnn.enabled = True
    #cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    #model = UNet(in_chns=3, class_num=cfg['nclass'])
    
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    #optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = -1 #int(os.environ["LOCAL_RANK"])
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #model.cuda()
    model.to(device)

    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
    #                                                  output_device=local_rank, find_unused_parameters=False)

    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).to(device) #.cuda(local_rank)
        #criterion_l = nn.CrossEntropyLoss().to(device) #.cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).to(device) #.cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])


    criterion_u = nn.CrossEntropyLoss(reduction='none').to(device) #.cuda(local_rank)
    
    #class_weights = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])#, 10.0, 10.0])
    #criterion_ce = nn.CrossEntropyLoss(weight= class_weights.to(device), reduction='mean')#**cfg['criterion']['kwargs'])
    #criterion_ce = nn.CrossEntropyLoss(reduction='mean').to(device)#, ignore_index=0)
    #criterion_dice = DiceLoss(n_classes=cfg['nclass']).to(device)

    trainset_u = EndovisDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = EndovisDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = EndovisDataset(cfg['dataset'], cfg['data_root'], 'val')

    #trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainsampler_l = torch.utils.data.RandomSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    #trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainsampler_u = torch.utils.data.RandomSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    trainsampler_u_mix = torch.utils.data.RandomSampler(trainset_u)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u_mix)
    #valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valsampler = torch.utils.data.RandomSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        #trainloader_l.sampler.set_epoch(epoch)
        #trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        """
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
        """
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            #img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_x, mask_x = img_x.to(device), mask_x.to(device)
            img_u_w = img_u_w.to(device) #.cuda()
            #img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.to(device), img_u_s2.to(device), ignore_mask.to(device)
            #img_u_s1, img_u_s2 = img_u_s1.to(device), img_u_s2.to(device)
            #cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.to(device), cutmix_box2.to(device)
            img_u_w_mix = img_u_w_mix.to(device) #.cuda()
            #img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.to(device), img_u_s2_mix.to(device)
            ignore_mask_mix = ignore_mask_mix.to(device) #.cuda()

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

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]            

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            #mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            #mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            #-----------------------------------------------
            #pred_x = pred_x.argmax(1).float()
            #------------------------------------------------
            
            if i==0 or i==80:
                visualize(img_x, pred_x, mask_x, 'x')
                #visualize(img_u_w, pred_u_w, mask_u_w, 'u_w')
                visualize(img_u_s1, pred_u_s1, mask_u_w_cutmixed1, 'u_s1')
                visualize(img_u_s2, pred_u_s2, mask_u_w_cutmixed2, 'u_s2')
                visualize(img_u_w, pred_u_w_fp, mask_u_w, 'u_w_fp')
            
            loss_x = criterion_l(pred_x, mask_x)
            #loss_x = criterion_ce(pred_x, mask_x)
            #loss_x = (criterion_ce(pred_x, mask_x) + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0
            #loss_x = criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())
            
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            """
            
            loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float())
            """
            """
            ce_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            ce_u_s1 = ce_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            ce_u_s1 = ce_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s1 = (ce_u_s1 + criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float()))/2.0
            """ 
                                      
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()
            """
            loss_u_s2 = criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']).float())
            
            loss_u_w_fp = criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float(),
                                         ignore=(conf_u_w < cfg['conf_thresh']).float())
            """
            """
            ce_u_s2 = criterion_ce(pred_u_s2, mask_u_w_cutmixed2)
            ce_u_s2 = ce_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            ce_u_s2 = ce_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_s2 = (ce_u_s2 + criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']).float()))/2.0
            
            ce_u_w_fp = criterion_ce(pred_u_w_fp, mask_u_w)
            ce_u_w_fp = ce_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            ce_u_w_fp = ce_u_w_fp.sum() / (ignore_mask != 255).sum().item()
            
            loss_u_w_fp = (ce_u_w_fp + criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float(),
                                         ignore=(conf_u_w < cfg['conf_thresh']).float()))/2.0
            """
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp *0.5) / 2.0 

            #torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            """
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())
            """
            mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            #optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_mask_ratio.avg)) 
                #logger.info('Iters: {:}, Total loss: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                #            '{:.3f}'.format(i, total_loss.avg, total_loss_s.avg,
                #                            total_loss_w_fp.avg, total_mask_ratio.avg)) 
        
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        """
        model.eval()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        j=0
        with torch.no_grad():
            for img, mask, id in loader:
                j+=1
                
                img = img.to(device) #cuda()
                pred = model(img).argmax(dim=1)
                            
                intersection, union, target = \
                    intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

                if epoch %10==0 and i%40==0:
                    im = img.cpu().numpy()
                    pre = pred.cpu().numpy()
                    mas = mask.cpu().numpy()
                    im=im.transpose(2,3,1,0).squeeze()
                    pre=pre.transpose(1,2,0)
                    mas = mas.transpose(1,2,0)
                    fig, ax = plt.subplots(1,3)
                    ax[0].set_title('Image',fontsize = 5)
                    ax[0].imshow(im)
                    ax[0].axis('off')

                    ax[1].set_title('Ground Truth',fontsize = 5)
                    ax[1].imshow(np.zeros(mas.shape),cmap='gray')
                    #mask[mask==0]=np.nan
                    ax[1].imshow(mas,cmap='Set1',norm=Normalize(1,10))
                    ax[1].axis('off')

                    ax[2].set_title('Pred',fontsize = 5)
                    ax[2].imshow(np.zeros(pre.shape),cmap='gray')
                    #pre[pre==0]=np.nan
                    ax[2].imshow(pre,cmap='Set1',norm=Normalize(1,10))
                    ax[2].axis('off')

                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0, hspace=0)

                    plt.savefig('output/frame_post.png', dpi=400, bbox_inches='tight')
                    plt.close()
                    

                reduced_intersection = torch.from_numpy(intersection).to(device) #cuda()
                reduced_union = torch.from_numpy(union).to(device) #cuda()

                intersection_meter.update(reduced_intersection.cpu().numpy())
                union_meter.update(reduced_union.cpu().numpy())

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
        mIOU = np.mean(iou_class)

        return mIOU, iou_class"""
        mIoU_all, mIoU_inst, iou_class, IoU, IoU1, mIoU  = evaluate(model, valloader, eval_mode, cfg, device, epoch)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
              logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> IoU: {:.2f}'.format(eval_mode, IoU))
            logger.info('***** Evaluation {} ***** >>>> mIoU: {:.2f}'.format(eval_mode, mIoU))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU_Instruments: {:.2f}\n'.format(eval_mode, mIoU_inst))
            
            writer.add_scalar('eval/IoU', IoU, epoch)
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_inst', mIoU_inst, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        """
        model.eval()
        dice_class = [0] * 9
        
        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.to(device), mask.to(device)

                h, w = img.shape[-2:]
                img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

                img = img.permute(1, 0, 2, 3)
                
                pred = model(img)
                
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg['nclass']):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)
        
        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
            
            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        """
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        

if __name__ == '__main__':
    main()
