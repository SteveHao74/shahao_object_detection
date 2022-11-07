import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
from torchsummary import summary
from torchvision import datasets, transforms
from engine import train_one_epoch,evaluate
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from xml.dom.minidom import parse
from engine import train_one_epoch, evaluate
import utils
import argparse
import tensorboardX
from sha_data_loader import coco_DataSet,RandomHorizontalFlip,Compose,ToTensor


def get_object_detection_model(num_classes=23):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # replace the classifier with a new one, that has num_classes which is user-defined
    # num_classes =22  # 3 class (mark_type_1，mark_type_2) + background
 
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
 
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model




# def train_per_epoch(epoch, net, device, train_data, optimizer, batches_per_epoch):
#     results = {
#         'loss': 0,
#         'losses': {
#         }
#     }

#     output=net.train()

# def eval_per_epoch():


def train_model(args):
    coco_root = "/media/shahao/F07EE98F7EE94F42/win_stevehao/ZJU/course/模式识别/人车检测/train_dataset"
    save_folder = os.path.join(args.logdir, args.description)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)
    device = torch.device("cuda")
    # 3 classes, mark_type_1，mark_type_2，background
    num_classes = 23
    # use our dataset and defined transformations
    train_transform = Compose([ToTensor()])#RandomHorizontalFlip(0.5),
    test_transform = Compose([ToTensor()])
    tmp_dataset = coco_DataSet(coco_root,train_transform )
    tmp_dataset_eval = coco_DataSet(coco_root,test_transform)
    # split the dataset in train and test set
    indices = torch.randperm(len(tmp_dataset)).tolist()
    dataset = torch.utils.data.Subset(tmp_dataset, indices[:int(args.split*len(tmp_dataset))])
    dataset_eval = torch.utils.data.Subset(tmp_dataset_eval, indices[int(args.split*len(tmp_dataset)):])

    # define training and validation data loaders
    # 在jupyter notebook里训练模型时num_workers参数只能为0，不然会报错，这里就把它注释掉了
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=1, shuffle=False,num_workers=args.num_workers,
        collate_fn=utils.collate_fn)
    if args.resume :
        retore_path = "/home/shahao/Project/object_detection/model/car_detection/epoch_50_iou_"
        model = torch.load(retore_path)
        start_epoch = 51
    else :
        # get the model using our helper function
        model = get_object_detection_model(num_classes)  # 或get_object_detection_model(num_classes)        
        start_epoch = 0
    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # SGD
    optimizer = torch.optim.SGD(params, lr=0.000001,
                                momentum=0.00001, weight_decay=0.0005)#params, lr=0.0005,momentum=0.9,
    # and a learning rate scheduler
    # cos学习率train_model
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    # let's train it for   epochs
    for epoch in range(start_epoch,args.epochs):
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,lr_scheduler=lr_scheduler)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset    
        coco_evaluator=evaluate(model, data_loader_eval, device=device)    
        
        # if  epoch % 3 == 0:#iou > best_iou or
        torch.save(model, os.path.join(save_folder, 'epoch_%02d_iou_' % epoch))#%0.2f
        print('coco_evaluator',coco_evaluator)
        print('==================================================')
        print('')

    print("That's it!")

if __name__ == '__main__':

    MODEL_SAVE_PATH = "/home/shahao/Project/object_detection/model"
    INPUT_DATA_PATH = "/media/shahao/F07EE98F7EE94F42/win_stevehao/ZJU/course/模式识别/人车检测/train_dataset" 
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--resume', type=str, default=False,
                        help='to resume the interrupted model training')
    # Network
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=300,
                        help='Input image size for the network')
    parser.add_argument('--output-size', type=int, default=300,
                        help='output image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str,default="jacquard",
                        help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, default=INPUT_DATA_PATH,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=292,
                        help='Batches per Epoch')#99,1287,1296
    parser.add_argument('--val-batches', type=int, default=100, help='Validation Batches')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')

    # Logging etc.
    parser.add_argument('--description', type=str, default='car_detection',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default=MODEL_SAVE_PATH,
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=52147,
                        help='Random seed for numpy')

    args = parser.parse_args()

    train_model(args)