import os
import json
import numpy as np
import numpy
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from PIL import Image
from sha_data_loader import coco_DataSet,RandomHorizontalFlip,Compose,ToTensor
MODEL_SAVE_PATH = "/home/shahao/Project/object_detection/model/car_detection/epoch_35_iou_"
TEST_SET_PATH = "/media/shahao/F07EE98F7EE94F42/win_stevehao/ZJU/course/模式识别/人车检测/test_dataset/test_images" 
VISUALIZATION_PATH = "/home/shahao/Project/object_detection/shahao_detection/test_visualization"

class_dict={"van":1, 'Truck':2, 'Pickup':3, 'Car':4, 'MediumBus':5, 'Pedestrian':6, 'BiCyclist':7, 'TricycleClosed':8, 'OtherCar':9,
     'PersonSitting':10, 'LightTruck':11, 'TricycleOpenHuman':12, 'HeavyTruck':13, 'MMcar':14, 'EngineTruck':15, 'LargeBus':16, 
     'TricycleOpenMotor':17, 'Bike':18, 'CampusBus':19, 'Machineshop':20, 'MotorCyclist':21, 'Motorcycle':22
    }
class_dict_convert = {v:k   for k,v in class_dict.items()}

def visualization(img,id,box,label):
    
    ax = plt.gca()
    # 默认框的颜色是黑色，第一个参数是左上角的点坐标
    # 第二个参数是宽，第三个参数是长
    ax.add_patch(plt.Rectangle((box["xmin"], box["ymin"]), box["xmax"]-box["xmin"] ,box["ymax"]-box["ymin"], color="blue", fill=False, linewidth=1))
    # 第三个参数是标签的内容
    # bbox里面facecolor是标签的颜色，alpha是标签的透明度
    ax.text(box["xmin"], box["ymin"], label, fontsize = 8, bbox={'facecolor':'blue', 'alpha':0.4})
    
    # plt.show()

def get_test_img():
    input_dict_list = []
    for root,dirs,files in os.walk(TEST_SET_PATH):
        for file in files:
            input_dict ={}
            path = os.path.join(root,file)
            # print(path)
            input_dict["img_id"]=path[-9:]
            input_dict["img"] = np.array(Image.open(path))#.convert('RGB')
            # print(input_dict["img"].size)
            input_dict_list.append(input_dict)
    # print(img_paths)
    return input_dict_list
def main():
    
    device = torch.device("cuda")
    input_dict_list = get_test_img()
    model=torch.load(MODEL_SAVE_PATH)
    model = model.to(device)

    total_annotations = {}
    total_annotations["annotations"] = []
    shahao = []
    for input_dict in input_dict_list:
        input = input_dict["img"]
        path  = input_dict["img_id"]
        transf = transforms.ToTensor()
        input_tensor = transf(input)
        input_tensor = input_tensor.to(device)
        print(input_tensor.size())
        input_tensor=input_tensor.reshape(1,3,720,1280)
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            output = model(input_tensor)
            output = output[0] 
            # type(output)
            print(output)
        # image_wise_dict ={}
        plt.clf()
        plt.cla() 
        plt.imshow(input)
        for i in range(len(output["labels"].cpu().numpy().tolist())):
            image_wise_dict ={}
            image_wise_dict["filename"] = "test_images\\"+path 
            box_list = output["boxes"].cpu().numpy().tolist()[i]
            image_wise_dict["box"] = {"xmin":box_list[0],"ymin":box_list[1],"xmax":box_list[2],"ymax":box_list[3]}
            image_wise_dict["label"] = class_dict_convert[output["labels"].cpu().numpy().tolist()[i]]
            image_wise_dict["conf"] =  output["scores"].cpu().numpy().tolist()[i]
            total_annotations["annotations"].append(image_wise_dict)

            visualization(input,path,image_wise_dict["box"],image_wise_dict["label"])

        plt.savefig(os.path.join(VISUALIZATION_PATH,path+".png"))

            # shahao.append(image_wise_dict)
            # import pdb; pdb.set_trace()
            # print(total_annotations)
            # print(image_wise_dict)


    
    #         _, preds = torch.max(outputs, 1)
    #         annotations.append({"filename":paths[0], "label": preds.cpu().numpy().tolist()[0]})
    # key = {"annotations":annotations}

    with open('submit_35.json','w') as f:
        json.dump(total_annotations, f, sort_keys=False, indent =4)

    
    
if __name__ == "__main__":
    main()