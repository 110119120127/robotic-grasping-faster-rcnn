# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from imageio import imread
#from scipy.misc import imread
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
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

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

if __name__ == '__main__':

  cuda = True
  cfg.USE_GPU_NMS = cuda
  cfg_file = 'cfgs/vgg16.yml'
  cfg_from_file(cfg_file)
  dataset = 'grasp'
  net = 'vgg16'
  load_dir = './models'
  image_dir = 'images_grasp'
  class_agnostic = True
  checksession = 1
  checkepoch = 5
  checkpoint = 899
  
  input_dir =  load_dir + "/" + net + "/" + dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))

  grasp_classes = np.asarray(['__background__',
                                'bolt', 'hammer', 'scissors', 'tape'])                                

  grasp_poses = np.asarray(['__background__',  # always index 0
                       'bin01', 'bin02', 'bin03', 'bin04', 'bin05', 'bin06', \
                       'bin07', 'bin08', 'bin09', 'bin10', 'bin11', 'bin12', \
                       'bin13', 'bin14', 'bin15', 'bin16', 'bin17', 'bin18', \
                       'binAll'])            
                  
                                
  # initilize the network here.
  fasterRCNN = vgg16(grasp_classes, grasp_poses, pretrained=False, class_agnostic=class_agnostic)

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))
  
  with torch.no_grad():   
      # initilize the tensor holder here.
      im_data = torch.FloatTensor(1)
      im_info = torch.FloatTensor(1)
      num_boxes = torch.LongTensor(1)
      gt_boxes = torch.FloatTensor(1)

      # ship to cuda
      if cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

  cfg.CUDA = cuda
  fasterRCNN.cuda()
  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = True

  imglist = os.listdir(image_dir) # add all file name of images in a list
  num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))

  while (num_images >= 0):
      total_tic = time.time()
      
      num_images -= 1
      if num_images == -1:
        break
        
      # Get image from the webcam
      im_file = os.path.join(image_dir, imglist[num_images])
      # im = cv2.imread(im_file)
      im_in = np.array(imread(im_file))
            
      
      if len(im_in.shape) == 2: # grayscale image, wxh
        im_in = im_in[:,:,np.newaxis] # extent to wxhx1
        im_in = np.concatenate((im_in,im_in,im_in), axis=2) # now become wxhx3
      
      # rgb -> bgr
      im = im_in[:,:,::-1] # # all items in the array, reversed

      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      with torch.no_grad():
              im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
              im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
              gt_boxes.resize_(1, 1, 6).zero_()
              num_boxes.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, ps_prob, RCNN_loss_ps, rois_pose = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      
      scores_ps = ps_prob.data

      if cfg.TEST.BBOX_REG: # Test using bounding-box regressors, True
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data   # (1, 300, 4)
          
          # True, set in lib/model/utils
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if class_agnostic: # our case
                if cuda > 0:
                    # (300, 4) 
                    # BBOX_NORMALIZE_STDS=(0.1, 0.1, 0.2, 0.2), BBOX_NORMALIZE_MEANS=(0.0, 0.0, 0.0, 0.0)
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)  # (1, 300, 4)
            else:
                if cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4 * len(grasp_classes))
          # boxes: RoIs output from RPN, in image coordinates  
          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          # Clip boxes to image boundaries
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]    # im_scales[0] = 1.25
    
      # (1,300,5) --> (300,5). 5: classes
      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      
      scores_ps = scores_ps.squeeze()
      
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      
      if vis:   # vis = True
          im2show = np.copy(im)
      

      daset_classes = grasp_classes                 
      
      # start from ind 1, ignore the bg cls
      for j in xrange(1, len(daset_classes)):
          # scores.shape = (300,5), 5: len(daset_classes)
          inds = torch.nonzero(scores[:,j]>thresh).view(-1) # thresh=0.05
          # if there is det
          # scores[:,j].shape = (300,)
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            ps_scores = scores_ps[inds]
            ps_scores_inds = torch.max(ps_scores,1)[1]
            #print('ps_scores_inds.shape: {}'.format(ps_scores_inds.shape))
            
            #print('ps_scores.shape: {}'.format(ps_scores.shape))
            #print('cls_scores.shape: {}'.format(cls_scores.shape))
            # sort along axis 0, in descending order (fag = True)
            _, order = torch.sort(cls_scores, 0, True)
            if class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            # cls_boxes for each class contains N number of bboxes
            #print('cls_boxes.shape: {}'.format(cls_boxes.shape))
            # cls_scores.unsqueeze(1): ex (32,) --> (32,1)
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), ps_scores_inds.unsqueeze(1).float()), 1)
            # cls_dets[0,:]: (x1,y1,x2,y2,cls)
            #print('cls_dets.shape: {}'.format(cls_dets.shape))
                        
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            #print('cls_dets[0,:]: {}'.format(cls_dets[0,:]))
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            #print('keep: {}'.format(keep))
            cls_dets = cls_dets[keep.view(-1).long()]
            #print('cls_dets.shape after NMS: {}'.format(cls_dets.shape))
            #print('cls_dets: {}'.format(cls_dets))
            
            if vis:
              # only show bboxes having class score > 0.5  
              im2show = vis_detections(im2show, daset_classes[j], cls_dets.cpu().numpy(), 0.5)

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic
        
      print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images+1, len(imglist), detect_time, nms_time))

      if vis:
          print('imglist: {}'.format(imglist[num_images]))
          # [:-4]: obmit the last 4 characters ('.jpg')
          result_path = os.path.join(image_dir, imglist[num_images][:-4] + "_det.jpg")
          cv2.imwrite(result_path, im2show)

      
