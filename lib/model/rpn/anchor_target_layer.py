from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_overlaps, bbox_transform_batch, bbox_iog
import math
import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        #import pdb
        '''
        anchors_array = [[   9.,   22.],
               [  11.,   28.],
               [  14.,   33.],
               [  16.,   40.],
               [  19.,   46.],
               [  21.,   52.],
               [  24.,   59.],
               [  27.,   66.],
               [  30.,   74.],
               [  34.,   83.],
               [  38.,   91.],
               [  41.,  100.],
               [  45.,  109.],
               [  49.,  119.],
               [  52.,  128.],
               [  57.,  138.],
               [  61.,  148.],
               [  65.,  159.],
               [  70.,  170.],
               [  75.,  182.],
               [  80.,  195.],
               [  85.,  208.],
               [  92.,  225.],
               [  99.,  242.],
               [ 107.,  260.],
               [ 116.,  282.],
               [ 125.,  305.],
               [ 137.,  335.],
               [ 150.,  366.],
               [ 170.,  416.],
               [ 188.,  459.],
               [ 209.,  511.],
               [ 239.,  582.]]  
                                                                                                           
        #_anchors = torch.zeros(len(anchors_array), 4)
        for i in range(len(anchors_array)):
            self._anchors[i][0]=-((anchors_array[i][0]-1)/2)
            self._anchors[i][2]=((anchors_array[i][0]-1)/2)+anchors_array[i][0]%2
            self._anchors[i][1]=-((anchors_array[i][1]-1)/2)
            self._anchors[i][3]=((anchors_array[i][1]-1)/2)+anchors_array[i][1]%2
        '''

        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        #ped_boxes=torch.zeros(gt_boxes.shape[0], int(max(torch.sum(gt_boxes[:,:,4]==1,1))),5).cuda()
        ped_boxes=torch.zeros(gt_boxes.shape[0], gt_boxes.shape[1],5).cuda()
        ignore_boxes=torch.zeros(gt_boxes.shape[0], int(max(torch.sum(gt_boxes[:,:,4]==2,1))),4).cuda()
        hard_boxes=torch.zeros(gt_boxes.shape[0], int(max(torch.sum(gt_boxes[:,:,4]==3,1))),4).cuda()

        for i in range(gt_boxes.shape[0]):
            index_hard=0
            index_ignore=0
            for j in range(gt_boxes.shape[1]):
                if gt_boxes[i,j,4]==2:
                    ignore_boxes[i,index_ignore,:]=gt_boxes[i,j,:4]
                    index_ignore+=1
                elif gt_boxes[i,j,4]==3:
                    hard_boxes[i,index_hard,:]=gt_boxes[i,j,:4]
                    index_hard+=1
                else:
                    ped_boxes[i,j,:]=gt_boxes[i,j,:]

        #print('gt:  ' + str(gt_boxes))
        ##print('ped:  ' + str(ped_boxes))
        #print('ignore:  ' + str(ignore_boxes))
        ##print('hard:  ' + str(hard_boxes))
        #import pdb
        #pdb.set_trace()
        #print('ped_boxes:  ' + str(ped_boxes))
        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = ped_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)
        self._anchors = self._anchors.type_as(ped_boxes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4).cuda()
        all_anchors = all_anchors.view(K * A, 4)

        #import pdb
        #pdb.set_trace()

        total_anchors = int(K * A)

        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = ped_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = ped_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = ped_boxes.new(batch_size, inds_inside.size(0)).zero_()


        overlaps = bbox_overlaps_batch(anchors, ped_boxes)

        #import pdb
        #pdb.set_trace()

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)
        #print(gt_max_overlaps)

        #import pdb
        #pdb.set_trace()

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0


        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)
        #print(sum_fg)
        #print(sum_bg)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(ped_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(ped_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1
            #pdb.set_trace() 
#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels[i] == 1).int())

            #print('num_fg:  ' + str(torch.sum((labels[i] == 1).int())))
            #print('num_bg:  ' + str(num_bg))

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(ped_boxes).long()

                labels[i][bg_inds]=-1

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(ped_boxes).long()

                import time
                t0=time.time()
                #pdb.set_trace()
                '''
                for j in range(int(num_bg)):
                    while iob(anchors[rand_num[rand_num.shape[0]-j-1]], ignore_boxes[i]) or iou(anchors[rand_num[rand_num.shape[0]-j-1]], hard_boxes[i]):
                        print('swap')
                        #swap(rand_num(rand_num.shape[0]-j-1], head)
                        temp1=int(rand_num[rand_num.shape[0]-j-1])
                        temp2=int(rand_num[head])
                        rand_num[rand_num.shape[0]-j-1]=temp2
                        rand_num[head]=temp1
                        head+=1
                '''
                #print(anchors[rand_num].shape)
                #print(hard_boxes[i].shape)
                #import pdb
                #pdb.set_trace()

                #import pdb
                #pdb.set_trace()
                if hard_boxes[i].numel()!=1:
                    hard_overlaps = bbox_overlaps(anchors[rand_num], hard_boxes[i])
                    rand_num = rand_num[torch.sum(hard_overlaps, 1)==0]
                if ignore_boxes[i].numel()!=1:
                    ignore_overlaps = bbox_iog(anchors[rand_num], ignore_boxes[i])
                    rand_num = rand_num[torch.sum(ignore_overlaps, 1)==0]

                t1=time.time()
                #print('ignore and hard:  ' + str(t1-t0))
                #disable_inds = bg_inds[rand_num[num_bg:]]
                labels[i][rand_num[:num_bg]] = 0

                #print(torch.sum((labels[i] == 1).int()))
                #print(torch.sum((labels[i] == 0).int()))
 
        offset = torch.arange(0, batch_size)*ped_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, ped_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            #num_examples = torch.sum(labels[i] >= 0)
            num_examples = torch.sum(labels[i] >= 0).item()
            positive_weights = 1.0 / num_examples
            negative_weights = 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        
        outputs.append(bbox_outside_weights)
        #import pdb
        #pdb.set_trace()

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
