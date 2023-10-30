import argparse
import logging
import os
import pprint

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.endovis import EndovisDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import random

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--device', default='cuda:3')

def visualize(image, pred, mask, name):
    image_c = image.detach().cpu().numpy()
    pred_c = pred.detach().cpu().numpy()
    mask_c = mask.detach().cpu().numpy()
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
    ax[1].imshow(mask_c,cmap='Set1',norm=Normalize(1,10))
    ax[1].axis('off')

    ax[2].set_title('Pred',fontsize = 5)
    ax[2].imshow(np.zeros(pred_c.shape),cmap='gray')
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

    rank, world_size = 0, 1

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    model = DeepLabV3Plus(cfg)
    
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model.to(device)
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).to(device) #.cuda(local_rank)
        #criterion_l = nn.CrossEntropyLoss().to(device) #.cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).to(device) #.cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])


    criterion_u = nn.CrossEntropyLoss(reduction='none').to(device) #.cuda(local_rank)
    valset = EndovisDataset(cfg['dataset'], cfg['data_root'], 'val')

    valsampler = torch.utils.data.RandomSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU_all, mIoU_inst, iou_class, IoU, IoU1, mIoU = evaluate(model, valloader, eval_mode, cfg, device, epoch, 'eval')

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

if __name__ == '__main__':
    main()