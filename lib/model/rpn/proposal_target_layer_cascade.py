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
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch, bbox_overlaps, bbox_iog
import pdb
import math

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes):

        #import pdb
        #pdb.set_trace()
        #ped_boxes=torch.zeros(gt_boxes.shape[0], int(max(torch.sum(gt_boxes[:,:,4]==1,1))),5).type_as(gt_boxes)
        ped_boxes=torch.zeros(gt_boxes.shape[0], gt_boxes.shape[1],5).type_as(gt_boxes)
        ignore_boxes=torch.zeros(gt_boxes.shape[0], int(max(torch.sum(gt_boxes[:,:,4]==2,1))),4).type_as(gt_boxes)
        hard_boxes=torch.zeros(gt_boxes.shape[0], int(max(torch.sum(gt_boxes[:,:,4]==3,1))),4).type_as(gt_boxes)

        #pdb.set_trace()
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

        #import pdb
        #pdb.set_trace()

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(ped_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(ped_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(ped_boxes)

        ped_boxes_append = ped_boxes.new(ped_boxes.size()).zero_()
        ped_boxes_append[:,:,1:5] = ped_boxes[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, ped_boxes_append], 1)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        #import pdb
        #pdb.set_trace()

        labels, rois, bbox_targets, bbox_inside_weights, gt_rois = self._sample_rois_pytorch(
            all_rois, ped_boxes, ignore_boxes, hard_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        #import pdb
        #pdb.set_trace()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, gt_rois

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            try:
                if clss[b].sum() == 0:
                    continue
                inds = torch.nonzero(clss[b] > 0).view(-1)
            except:
                import pdb
                pdb.set_trace()
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        #import pdb
        #pdb.set_trace()

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, ped_boxes, ignore_boxes, hard_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x ped_boxes)

        #import pdb
        #pdb.set_trace()

        #import pdb
        #pdb.set_trace()
        #print(int(torch.sum(torch.sum(all_rois[:,:2000,1:]==0,2)==4)))
        overlaps = bbox_overlaps_batch(all_rois, ped_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)


        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*ped_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        #labels = ped_boxes[:,:,4].contiguous().view(-1).index(offset.view(-1))\
        #                                                    .view(batch_size, -1)
        labels = ped_boxes[:,:,4].contiguous().view(-1).index((offset.view(-1), ))\
                                                            .view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()
            #print(fg_num_rois)
            #print(fg_inds)
            #print('ped number:  ' + str((sum(ped_boxes[0,:,4]==1))))


            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()


            if fg_num_rois > 0 and bg_num_rois > 0:
                #print('enter both')
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault. 
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(ped_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error. 
                # We use numpy rand instead. 
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                #import pdb
                #pdb.set_trace()
                #rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                #rand_num = torch.from_numpy(rand_num).type_as(ped_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds)).type_as(ped_boxes).long()

                if hard_boxes[i].numel()!=1:
                    hard_overlaps = bbox_overlaps(all_rois[i,rand_num,:4], hard_boxes[i])
                    rand_num = rand_num[torch.sum(hard_overlaps, 1)==0]
                if ignore_boxes[i].numel()!=1:
                    ignore_overlaps = bbox_iog(all_rois[i,rand_num,:4], ignore_boxes[i])
                    rand_num = rand_num[torch.sum(ignore_overlaps, 1)==0]

                if rand_num.shape[0]>=int(bg_rois_per_this_image):
                    bg_inds = rand_num[:bg_rois_per_this_image]
                    #print('if')
                else:
                    bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                               (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
                    rand_num = torch.from_numpy(np.random.permutation(bg_inds)).type_as(ped_boxes).long()

                    if rand_num.shape[0]>=int(bg_rois_per_this_image):
                        bg_inds = rand_num[:bg_rois_per_this_image]
                        #print('else')
                    else:
                        bg_inds = torch.zeros(bg_rois_per_this_image).type_as(ped_boxes).long()
                        bg_inds[:rand_num.shape[0]] = rand_num
                        bg_inds[rand_num.shape[0]:] = rand_num[:bg_rois_per_this_image-rand_num.shape[0]]
                        #import pdb
                        #pdb.set_trace()
                
                #print('rcnn ignore and hard:  ' + str(t1-t0))
            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(ped_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                #rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                #rand_num = torch.from_numpy(rand_num).type_as(ped_boxes).long()

                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0

                rand_num = torch.from_numpy(np.random.permutation(bg_inds)).type_as(ped_boxes).long()

                if hard_boxes[i].numel()!=1:
                    hard_overlaps = bbox_overlaps(all_rois[i,rand_num,:4], hard_boxes[i])
                    rand_num = rand_num[torch.sum(hard_overlaps, 1)==0]
                if ignore_boxes[i].numel()!=1:
                    ignore_overlaps = bbox_iog(all_rois[i,rand_num,:4], ignore_boxes[i])
                    rand_num = rand_num[torch.sum(ignore_overlaps, 1)==0]
  
                if rand_num.shape[0]>=int(bg_rois_per_this_image):
                    bg_inds = rand_num[:bg_rois_per_this_image]
                    #print('if')
                else:
                    bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                               (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
                    rand_num = torch.from_numpy(np.random.permutation(bg_inds)).type_as(ped_boxes).long()
                    
                    if rand_num.shape[0]>=int(bg_rois_per_this_image):
                        bg_inds = rand_num[:bg_rois_per_this_image]
                        #print('else')
                    else:
                        #import pdb
                        #pdb.set_trace()
                        bg_inds = torch.zeros(bg_rois_per_this_image).type_as(ped_boxes).long()
                        bg_inds[:rand_num.shape[0]] = rand_num
                        bg_inds[rand_num.shape[0]:] = rand_num[:bg_rois_per_this_image-rand_num.shape[0]]
                #print('RCNN ignore and hard:  ' + str(t1-t0))
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            #print(torch.sum(torch.sum(ped_boxes[i],1)!=0))
            #print(fg_inds.shape[0])
            #import pdb
            #pdb.set_trace()
            #print(bg_inds)
            # Select sampled values from various arrays:
            if int(keep_inds.shape[0])<256:
                import pdb
                pdb.set_trace()

            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            gt_rois_batch[i] = ped_boxes[i][gt_assignment[i][keep_inds]]

        #import pdb
        #pdb.set_trace()

        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])

        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)
        #pdb.set_trace()
        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights, gt_rois_batch[:,:,:4]
