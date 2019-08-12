from include import *
import pandas as pd
from process.augmentation import *
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
PJ_DIR = os.path.join(current_dir, '..')
LIST_DIR = os.path.join(PJ_DIR ,'image_list')

DATA = {}
train_df = None
TRN_IMGS_DIR = None
TST_IMGS_DIR = None

def set_dirs(data_root_dir=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if data_root_dir is None:
        data_root_dir = os.path.join(current_dir, '..', '..', 'data')

    print("setting data_root_dir", data_root_dir)

    global DATA
    DATA['train_df'] = pd.read_csv(os.path.join(data_root_dir, 'train.csv'))
    DATA['TRN_IMGS_DIR'] = os.path.join(data_root_dir, 'train')
    DATA['TST_IMGS_DIR'] = os.path.join(data_root_dir, 'test')

print(TRN_IMGS_DIR, TST_IMGS_DIR)

def load_label_dict(label_list_path):
    f = open(label_list_path, 'r')
    lines = f.readlines()
    f.close()

    label_dict = {}
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        id = line[0]
        index = int(line[1])
        label_dict[id] = index

    return label_dict

def get_list(csv):
    bbox_data = pd.read_csv(csv)
    labelName = bbox_data['Image'].tolist()
    x0 = bbox_data['x0'].tolist()
    y0 = bbox_data['y0'].tolist()
    x1 = bbox_data['x1'].tolist()
    y1 = bbox_data['y1'].tolist()
    return labelName,x0,y0,x1,y1

def read_txt(txt):
    f = open(txt, 'r')
    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    list = []
    for line in lines:
        line = line.split(' ')
        list.append([line[0], int(line[1])])

    return list

def load_train_list(train_image_list_path = LIST_DIR + r'/train_image_list.txt'):
    f = open(train_image_list_path, 'r')
    lines = f.readlines()
    f.close()

    list = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        img_name = line[0]
        index = int(line[1])
        list.append([img_name, index])
    return list

def load_train_map(train_image_list_path = LIST_DIR + r'/train_image_list.txt'):
    f = open(train_image_list_path, 'r')
    lines = f.readlines()
    f.close()

    label_dict = {}
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        img_name = line[0]
        index = int(line[1])
        id = line[2]
        label_dict[img_name] = [index,id]

    return label_dict

def load_bbox_dict():

    bbox_dict = {}
    csv_origin = PJ_DIR + r'/bbox_model/se50_bbox.csv'
    print(csv_origin)
    _, x0_se50, y0_se50, x1_se50, y1_se50 = get_list(csv_origin)

    csv_origin = PJ_DIR + r'/bbox_model/se101_bbox.csv'
    print(csv_origin)
    labelName, x0_se101, y0_se101, x1_se101, y1_se101 = get_list(csv_origin)
    for (name, x0,y0,x1,y1,x0_,y0_,x1_,y1_) in zip(labelName, x0_se50,y0_se50,x1_se50,y1_se50,
                                                              x0_se101,y0_se101,x1_se101,y1_se101):
        bbox_dict[name] = [(x0+x0_)//2,(y0+y0_)//2,(x1+x1_)//2,(y1+y1_)//2]
    return bbox_dict

def load_pseudo_list(path = LIST_DIR + r'/pseudo_list.txt'):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    list = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        img_name = line[0]
        index = int(line[1])
        list.append([img_name, index])
    return list

def image_list2dict(image_list):
    dict = {}
    id_list = []
    for image,id in image_list:
        if id in dict:
            dict[id].append(image)
        else:
            dict[id] = [image]
            id_list.append(id)

    return dict,id_list

def load_CLASS_NAME():
    label_list_path = LIST_DIR + r'/label_list.txt'
    f = open(label_list_path, 'r')
    lines = f.readlines()
    f.close()

    label_dict = {}
    id_dict = {}
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        id = line[0]
        index = int(line[1])

        if index == -1:
            index = 5004

        label_dict[index] = id
        id_dict[id] = index

    return label_dict, id_dict
