from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./devices')
from devices.camera import RealSenseD400 
import threading  
from operator import itemgetter

import _init_paths
import os
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from imageio import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16


def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)
  
  
class PoseEst:  
    def __init__(self):
        self.cuda = True
        cfg.USE_GPU_NMS = self.cuda
        cfg_file = 'cfgs/vgg16.yml'
        cfg_from_file(cfg_file)
        dataset = 'grasp'
        net = 'vgg16'
        load_dir = './models'
        self.class_agnostic = True
        checksession = 1
        checkepoch = 5
        checkpoint = 899  
        
        input_dir =  load_dir + "/" + net + "/" + dataset
        if not os.path.exists(input_dir):
            raise Exception('There is no input directory for loading network from ' + input_dir)
        
        load_name = os.path.join(input_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))
        
        self.grasp_classes = np.asarray(['__background__',
                                    'bolt', 'hammer', 'scissors', 'tape'])
        self.grasp_poses = np.asarray(['__background__',  # always index 0
                           'bin01', 'bin02', 'bin03', 'bin04', 'bin05', 'bin06', \
                           'bin07', 'bin08', 'bin09', 'bin10', 'bin11', 'bin12', \
                           'bin13', 'bin14', 'bin15', 'bin16', 'bin17', 'bin18', \
                           'binAll']) 
        
        # initilize the network here.
        self.fasterRCNN = vgg16(self.grasp_classes,
                                self.grasp_poses,
                                pretrained=False,
                                class_agnostic=self.class_agnostic)
        self.fasterRCNN.create_architecture()
        
        print("load checkpoint %s" % (load_name))
        if self.cuda > 0:
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        
        print('load model successfully!')
        # pdb.set_trace()
        print("load checkpoint %s" % (load_name))
        
        with torch.no_grad():   
            # initilize the tensor holder here.
            self.im_data = torch.FloatTensor(1)
            self.im_info = torch.FloatTensor(1)
            self.num_boxes = torch.LongTensor(1)
            self.gt_boxes = torch.FloatTensor(1)    
        
        # ship to cuda
        if self.cuda > 0:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()     
        
        cfg.CUDA = self.cuda
        self.fasterRCNN.cuda()
        self.fasterRCNN.eval()
        
        self.thresh = 0.05
        self.vis = True                       
    
    def pose_est(self, im_in):
        # rgb -> bgr
        im = im_in[:,:,::-1] # # all items in the array, reversed
        
        blobs, im_scales = _get_image_blob(im)
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)
        
        with torch.no_grad():
            self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self.gt_boxes.resize_(1, 1, 6).zero_()
            self.num_boxes.resize_(1).zero_()
        
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, ps_prob, RCNN_loss_ps, rois_pose = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)  
        
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        scores_ps = ps_prob.data
        
        if cfg.TEST.BBOX_REG: # Test using bounding-box regressors, True
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data   # (1, 300, 4)
            # True, set in lib/model/utils
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic: # our case
                    if self.cuda > 0:
                        # (300, 4) 
                        # BBOX_NORMALIZE_STDS=(0.1, 0.1, 0.2, 0.2), BBOX_NORMALIZE_MEANS=(0.0, 0.0, 0.0, 0.0)
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                      + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4)  # (1, 300, 4)

            # boxes: RoIs output from RPN, in image coordinates  
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            # Clip boxes to image boundaries
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        
        pred_boxes /= im_scales[0]    # im_scales[0] = 1.25
        # (1,300,5) --> (300,5). 5: classes
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        scores_ps = scores_ps.squeeze()
        if self.vis:   # self.vis = True
            im2show = np.copy(im)    
        
        daset_classes = self.grasp_classes                 
        pose_lists = []
        pose_highest_lists = []
        pose_highest = []
        
        # # start from ind 1, ignore the bg cls
        for j in range(1, len(daset_classes)): 
            inds = torch.nonzero(scores[:,j]>self.thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                ps_scores = scores_ps[inds]
                ps_scores_max_values, ps_scores_inds = torch.max(ps_scores,1)

                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1),
                                    ps_scores_max_values.unsqueeze(1),
                                    ps_scores_inds.unsqueeze(1).float()), 1)
                
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                
                if self.vis:            
                    # only show bboxes having class score > 0.5  
                    im2show, pose_list = vis_detections(im2show, 
                                            daset_classes[j], 
                                            cls_dets.cpu().numpy(), thresh=0.5)
                    # pose_list: list of all bboxes for each class
                    if len(pose_list): 
                        if len(pose_list) > 1:
                          # sort all bboxes of 1 class according to angle score
                          pose_list.sort(key=itemgetter(5), reverse=True)
                        
                        pose_lists.append(pose_list)
                        #print('pose_lists: {}'.format(pose_lists))
                        # only keep the bbox having the highest angle score of each class
                        pose_highest_lists.append(pose_list[0])
                        print('pose_highest_lists: {}'.format(pose_highest_lists))
                        
                        if len(pose_highest_lists) > 1:
                          # sort all highest bboxes of all classes, according to angle score
                          pose_highest_lists.sort(key=itemgetter(5), reverse=True)
                        
                        # get the highest angle core bbox accross all classes, bboxes
                        pose_highest = pose_highest_lists[0]
                    
        print('pose_highest: {}'.format(pose_highest))                
        
        im2show_copy = np.copy(im2show)
        im2show_copy = im2show_copy.astype(np.uint8)
        im2showRGB = cv2.cvtColor(im2show_copy, cv2.COLOR_BGR2RGB)
        
        return pose_highest, im2show_copy, im2showRGB

if __name__ == '__main__':

    rs400 = RealSenseD400() 
    cam_thread = threading.Thread(target=rs400.start_stream, args=())
    cam_thread.daemon = True
    cam_thread.start()
    pe = PoseEst()
    
    while True:
        img = rs400.image
        pose, imgBRG, imgRGB = pe.pose_est(img)
        if pose:
            time.sleep(1.0)
        
        cv2.namedWindow("frame")
        cv2.imshow("frame", imgRGB)
        key = cv2.waitKey(100)
        if key in [27, ord('q'), ord('Q')]:
            break
        
    rs400.stop()        
    cv2.destroyAllWindows()  
            
    
    
    
    
    
                
                
                 
                
                
                
        
            
            
        
              
        
  
  
  
  
  
  
  
  
  


































  
  
