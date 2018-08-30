import math
import torch
from torch.autograd import Variable
from model.utils.config import cfg
from bbox_transform import bbox_transform_inv, bbox_overlaps

def IoG(box_a, box_b):                                                                                             
    inter_xmin = torch.max(box_a[0], box_b[0])                                                                     
    inter_ymin = torch.max(box_a[1], box_b[1])                                                                     
    inter_xmax = torch.min(box_a[2], box_b[2])                                                                     
    inter_ymax = torch.min(box_a[3], box_b[3])                                                                     
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)                                                               
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)                                                               
    I = Iw * Ih                                                                                                    
    G = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])                                                              
    return I / G                                                                                                   
	
def repgt(pred_boxes, gt_rois, rois_inside_ws):

    sigma_repgt = 0.9
    loss_repgt=torch.zeros(pred_boxes.shape[0]).cuda()                                                                                                                                                      
    for i in range(pred_boxes.shape[0]):                                                                                                                                                                                       
        boxes = Variable(pred_boxes[i,rois_inside_ws[i]!=0].view(int(pred_boxes[i,rois_inside_ws[i]!=0].shape[0])/4,4))     
        gt = Variable(gt_rois[i,rois_inside_ws[i]!=0].view(int(gt_rois[i,rois_inside_ws[i]!=0].shape[0])/4,4))              
        num_repgt = 0
        repgt_smoothln=0
        if boxes.shape[0]>0:
            overlaps = bbox_overlaps(boxes, gt)
            for j in range(overlaps.shape[0]):
                for z in range(overlaps.shape[1]):
                    if int(torch.sum(gt[j]==gt[z]))==4:
                        overlaps[j,z]=0
            max_overlaps, argmax_overlaps = torch.max(overlaps,1)
            for j in range(max_overlaps.shape[0]):
                if max_overlaps[j]>0:
                    num_repgt+=1
                    iog = IoG(boxes[j], gt[argmax_overlaps[j]])
                    if iog>sigma_repgt:
                        repgt_smoothln+=((iog-sigma_repgt)/(1-sigma_repgt)-math.log(1-sigma_repgt))
                    elif iog<=sigma_repgt:
                        repgt_smoothln+=-math.log(1-iog)
        if num_repgt>0:
            loss_repgt[i]=repgt_smoothln/num_repgt
			
    return loss_repgt			

def repbox(pred_boxes, gt_rois, rois_inside_ws):

    sigma_repbox = 0
    loss_repbox=torch.zeros(pred_boxes.shape[0]).cuda()

    for i in range(pred_boxes.shape[0]):
        
        boxes = Variable(pred_boxes[i,rois_inside_ws[i]!=0].view(int(pred_boxes[i,rois_inside_ws[i]!=0].shape[0])/4,4))
        gt = Variable(gt_rois[i,rois_inside_ws[i]!=0].view(int(gt_rois[i,rois_inside_ws[i]!=0].shape[0])/4,4))
 
        num_repbox = 0
        repbox_smoothln = 0
        if boxes.shape[0]>0:
            overlaps = bbox_overlaps(boxes, boxes)
            for j in range(overlaps.shape[0]):
                for z in range(overlaps.shape[1]):
                    if z>=j:
                        overlaps[j,z]=0
                    elif int(torch.sum(gt[j]==gt[z]))==4:
                        overlaps[j,z]=0

            iou=overlaps[overlaps>0]
            for j in range(iou.shape[0]):
                num_repbox+=1
                if iou[j]<=sigma_repbox:
                    repbox_smoothln+=-math.log(1-iou[j])
                elif iou[j]>sigma_repbox:
                    repbox_smoothln+=((iou[j]-sigma_repbox)/(1-sigma_repbox)-math.log(1-sigma_repbox))

        if num_repbox>0:
            loss_repbox[i]=repbox_smoothln/num_repbox
            
    return loss_repbox
				   
def repulsion(rois, box_deltas, gt_rois, rois_inside_ws, rois_outside_ws):                                             
														   
    deltas = Variable(box_deltas.view(rois.shape[0],256,4))                                                        
    rois_inside_ws = Variable(rois_inside_ws.view(rois.shape[0],256,4))                                            
    rois_outside_ws = Variable(rois_outside_ws.view(rois.shape[0],256,4))                                          
    if int(torch.sum(rois_outside_ws==rois_inside_ws))!=1024:                                                      
	import pdb                                                                                                 
	pdb.set_trace() 

    for i in range(rois.shape[0]):
	deltas[i] = deltas[i].view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()+ \
                     torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

    pred_boxes = bbox_transform_inv(rois[:,:,1:5], deltas, 2)

    loss_repgt = repgt(pred_boxes, gt_rois, rois_inside_ws)
    loss_repbox = repbox(pred_boxes, gt_rois, rois_inside_ws)

    return loss_repgt, loss_repbox
