# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""
import os

import torch
import torch.nn as nn

from resnet import resnet18, resnet34, resnet50, resnet152, resnet101, resnet200
# from bn_alexnet import bn_alexnet, bn_alex_deepclustering,rot_AlexNet,load_pretrained


def model_select(args):

    # ResNet-18
    if args.usenet == "resnet18":
        last_layer = nn.Linear(512, args.numof_classes)
        model = resnet18(pretrained=False, num_classes=args.numof_pretrained_classes)
        weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")

        # FractalDB pre-trained model
        if os.path.exists(weight_name):
            print("use pretrained model : %s" % weight_name)
            param = torch.load(weight_name, map_location=lambda storage, loc: storage)
            model.load_state_dict(param)
        # ImageNet pre-trained model
        elif args.dataset == "imagenet":
            print("use imagenet pretrained model")
            model = resnet18(pretrained=True)
        model.fc = last_layer

    # ResNet-34
    if args.usenet == "resnet34":
        last_layer = nn.Linear(512, args.numof_classes)
        model = resnet34(pretrained=False, num_classes=args.numof_pretrained_classes)
        weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")

        # FractalDB pre-trained model
        if os.path.exists(weight_name):
            print("use pretrained model : %s" % weight_name)
            param = torch.load(weight_name, map_location=lambda storage, loc: storage)
            model.load_state_dict(param)
        # ImageNet pre-trained model
        elif args.dataset == "imagenet":
            print("use imagenet pretrained model")
            model = resnet34(pretrained=True)
        model.fc = last_layer

    # ResNet-50
    if args.usenet == "resnet50":
        last_layer = nn.Linear(2048, args.numof_classes)
        model = resnet50(pretrained=False, num_classes=args.numof_pretrained_classes)
        weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")

        # FractalDB pre-trained model
        if os.path.exists(weight_name):
            print("use pretrained model : %s" % weight_name)
            param = torch.load(weight_name)
            model.load_state_dict(param)
        # ImageNet pre-trained model
        elif args.dataset == "imagenet":
            print("use imagenet pretrained model")
            model = resnet50(pretrained=True)
        model.fc = last_layer

    # ResNet-101
    if args.usenet == "resnet101":
        last_layer = nn.Linear(2048, args.numof_classes)
        model = resnet101(pretrained=False, num_classes=args.numof_pretrained_classes)
        weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")

        # FractalDB pre-trained model
        if os.path.exists(weight_name):
            print("use pretrained model : %s" % weight_name)
            param = torch.load(weight_name)
            model.load_state_dict(param)
        # ImageNet pre-trained model
        elif args.dataset == "imagenet":
            print("use imagenet pretrained model")
            model = resnet101(pretrained=True)
        model.fc = last_layer

    # ResNet-152
    if args.usenet == "resnet152":
        last_layer = nn.Linear(2048, args.numof_classes)
        model = resnet152(pretrained=False, num_classes=args.numof_pretrained_classes)
        weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")

        # FractalDB pre-trained model
        if os.path.exists(weight_name):
            print("use pretrained model : %s" % weight_name)
            param = torch.load(weight_name)
            model.load_state_dict(param)
        # ImageNet pre-trained model
        elif args.dataset == "imagenet":
            print("use imagenet pretrained model")
            model = resnet152(pretrained=True)
        model.fc = last_layer

    # ResNet-200
    if args.usenet == "resnet200":
        last_layer = nn.Linear(2048, args.numof_classes)
        model = resnet200(pretrained=False, num_classes=args.numof_pretrained_classes)
        weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")

        # FractalDB pre-trained model
        if os.path.exists(weight_name):
            print("use pretrained model : %s" % weight_name)
            param = torch.load(weight_name)
            model.load_state_dict(param)
        model.fc = last_layer

    return model
