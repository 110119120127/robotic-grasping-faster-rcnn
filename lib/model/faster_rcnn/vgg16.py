# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb


class vgg16(_fasterRCNN):
  def __init__(self, classes, poses, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    # init anchor, rpn, roi related layers
    _fasterRCNN.__init__(self, classes, class_agnostic, poses)

  def _init_modules(self):
    '''
    init the feature extractor, 2 shared fast rcnn fc layers, 2-specific-task fc layers '''
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
    
    # Remove the last fc, fc8 pre-trained for 1000-way ImageNet classification. Use the * operator to expand the list into positional arguments
    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer, maxpool_5
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3?:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier
    #self.RCNN_top_pose = vgg.classifier
    print(vgg.classifier)
    
    self.RCNN_top_pose = nn.Sequential(
        nn.Linear(in_features=7*7*512, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5))
    
    # 
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
    
    self.RCNN_ps_score = nn.Linear(4096, self.n_poses)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes) # *self.n_poses       

  def _head_to_tail(self, roi_pooled):
    
    roi_pooled_flat = roi_pooled.view(roi_pooled.size(0), -1)
    # fc7 of fast RCNN
    fc7 = self.RCNN_top(roi_pooled_flat)

    return fc7

  def _head_to_tail_pose(self, roi_pooled): 

    roi_pooled_flat = roi_pooled.view(roi_pooled.size(0), -1)
    fc7_pose = self.RCNN_top_pose(roi_pooled_flat) 

    return fc7_pose   

