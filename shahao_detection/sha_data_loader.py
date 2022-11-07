from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import os
import torch
import json
from PIL import Image
import numpy as np
#import transforms
from torchvision.transforms import functional as F
import random

#import matplotlib.pyplot as plt
coco_root = "/media/shahao/F07EE98F7EE94F42/win_stevehao/ZJU/course/模式识别/人车检测/train_dataset"


class coco_DataSet(Dataset):

    class_dict={"van":1, 'Truck':2, 'Pickup':3, 'Car':4, 'MediumBus':5, 'Pedestrian':6, 'BiCyclist':7, 'TricycleClosed':8, 'OtherCar':9,
     'PersonSitting':10, 'LightTruck':11, 'TricycleOpenHuman':12, 'HeavyTruck':13, 'MMcar':14, 'EngineTruck':15, 'LargeBus':16, 
     'TricycleOpenMotor':17, 'Bike':18, 'CampusBus':19, 'Machineshop':20, 'MotorCyclist':21, 'Motorcycle':22
    }
    def __init__(self, coco_root, transforms,train_set=True ):
        self.transforms = transforms
        self.annotations_json = os.path.join(coco_root,"train.json")
        self.image_root = os.path.join(coco_root,"train_images")
        assert os.path.exists(self.annotations_json), "{} file not exist.".format( self.annotations_json)
        json_file = open(self.annotations_json, 'r')
        self.coco_dict = json.load(json_file)
        self.bbox_image = {}                                                                 
        bbox_imag = self.coco_dict["annotations"]            
        last_file_id = None
        counter = -1
        for temp in bbox_imag:   
            pic_name = temp["filename"][-9:-4]#get the id of pic
            pic_id = int(pic_name)-1# filename begin with 00001
            if last_file_id != pic_id:
                counter +=1
                self.bbox_image[counter] = {"image_name":pic_name,"boxes":[],"labels":[],"labels":[],"iscrowd":[]}
            
            xmin = temp["box"]["xmin"]
            ymin = temp["box"]["ymin"]
            xmax = temp["box"]["xmax"]
            ymax = temp["box"]["ymax"]

            self.bbox_image[counter]["boxes"].append([xmin, ymin, xmax, ymax])                     
            self.bbox_image[counter]["labels"].append( self.class_dict[temp["label"]] )    
            self.bbox_image[counter]["iscrowd"].append( temp["occluded"])          #id
            last_file_id = pic_id
        for tem in range(len(self.bbox_image)):
            if self.bbox_image[tem]["boxes"]==[]:
                print("147",tem)

        # print("shahao",self.bbox_image[2571],len(self.bbox_image))

    def __len__(self):
        return len(self.bbox_image)

    def __getitem__(self,idx):
        # if  self.bbox_image.__contains__(idx) is False:
        #     return None,None
        instance_dict = self.bbox_image[idx]                           
        pic_name= instance_dict["image_name"]
        pic_path = os.path.join(self.image_root,pic_name+".jpg")
        image = Image.open(pic_path).convert("RGB")  
        shahao=np.array(instance_dict["boxes"])

        target = {}
        target["image_id"] = torch.tensor([idx])
        target["boxes"] = torch.from_numpy(shahao)
        target["labels"] = torch.as_tensor(instance_dict["labels"], dtype=torch.int64)
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.as_tensor(instance_dict["iscrowd"], dtype=torch.int64)
        # if self.bbox_image.__contains__(idx):
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def collate_fn(self,batch):
        return tuple(zip(*batch))

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target



class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox

        return image, target



data_transform = {
    "train": Compose([ToTensor()]),
    "val": Compose([ToTensor()])
}
cocodaset=coco_DataSet(coco_root,data_transform["train"])
# re_image = cocodaset.__getitem__(100)
print(cocodaset.bbox_image)
#print("re_target",re_target)

# dataloader = torch.utils.data.DataLoader(cocodaset, batch_size=2, shuffle=True,
#                         collate_fn= cocodaset.collate_fn   )  #lambda batch: tuple(zip(*batch)))
# for step, (batch_x, batch_y) in enumerate(dataloader):
#     print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
