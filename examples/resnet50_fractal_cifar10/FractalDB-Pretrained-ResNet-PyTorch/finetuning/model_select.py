# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""
import os

import torch
import torch.nn as nn

from resnet import *
#from bn_alexnet import bn_alexnet, bn_alex_deepclustering,rot_AlexNet,load_pretrained

def model_select(args):
	
	MODEL_ROOT = args.path2weight

	# ResNet-50
	if args.usenet == "resnet50":
		last_layer = nn.Linear(2048, args.numof_classes)
		model = resnet50(pretrained=False, num_classes=args.numof_pretrained_classes)
		# weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		weight_name = os.path.join(args.path2weight, "FractalDB-1000_res50_download.pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			print ("use imagenet pretrained model")
			model = resnet50(pretrained=True)
		model.fc = last_layer

	return model
