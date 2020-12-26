import cv2
import os
import glob
import shutil
import random
import mimetypes
import json
import base64
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from sympy import false, true

import utils
from detectron2.structures import BoxMode

def cityscapes2coco(img_filname, sub_dir, index):
    """Convert XML to json

        Args:
            json_path (str): Location of annotated XML file
        Returns:
            dictionary (dict): Annotation data in Dict format

        """


# ---------------------------------------------------------------------------
# --------------------------------   Script   -------------------------------
# ---------------------------------------------------------------------------

base_dir = os.path.join("./datasets/cityscapes/")
leftImg8bit_dir = os.path.join(base_dir, 'leftImg8bit')
gtFine_dir = os.path.join(base_dir, 'gtFine')
train_dir = 'train'
test_dir = 'test'
val_dir = 'val'

cities = []
for city in os.listdir(os.path.join(leftImg8bit_dir,train_dir)):
    if city != '.DS_Store':
        cities.append(city)

dict = {
    "images": [],
    "categories": [
        {
            "color": [
                107,
                142,
                35
            ],
            "id": 1,
            "isthing": 0,
            "name": "vegetation",
            "supercategory": "vegetation"
        },
    ],
    "annotations": [],
    "sem_seg_file_name": []
}

for dir_i, city in enumerate(cities):
    print(str(dir_i) + '/' + str(len(cities)) + ' BEGIN PROCCESSING - ' + cities[dir_i])

    # ディレクトリ（都市名）に対し画像を取得
    files = [n[:-15] for n in os.listdir(os.path.join(leftImg8bit_dir, train_dir, city))]

    # 画像に対しアノーテーションを作成
    for image_i, file in enumerate(files):
        # 対象のjsonファイルを読み込み
        json_path = os.path.join(gtFine_dir,train_dir,city,file+'gtFine_polygons.json')
        json_file = open(json_path, 'r')
        json_obj = json.load(json_file)

        # 対象の画像を読み込む
        img_path = os.path.join(leftImg8bit_dir, train_dir, city, file + 'leftImg8bit.png')
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 画像オブジェクトを生成
        meta = {
            "file_name": file + 'leftImg8bit.png',
            "sem_seg_file_name": file + 'gtFine_color.png',
            "id": image_i,
            "width": int(w),
            "height": int(h)
        }

        has_obj = false
        ann_objs = []

        # アンオーテーションを生成
        for object_i, obj in enumerate(json_obj['objects']):
            if obj['label'] == 'vegetation':
                ann_obj = {
                    "area": utils.calculate_area(obj['polygon']),
                    "bbox": utils.exrtract_dimensions(obj['polygon']),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [utils.extract_segmentations(obj['polygon'])],
                    "category_id": 1,
                    "image_id": image_i,
                    "id": '{0:04d}'.format(dir_i) + '{0:04d}'.format(image_i) + '{0:04d}'.format(object_i),
                    "iscrowd": 0
                }
                ann_objs.append(ann_obj)
                has_obj = true
        if has_obj:
            shutil.copyfile(os.path.join(leftImg8bit_dir, train_dir, city, file + 'leftImg8bit.png'),os.path.join(leftImg8bit_dir, 'semantic_train', file + 'leftImg8bit.png'))
            shutil.copyfile(os.path.join(gtFine_dir, train_dir, city, file + 'gtFine_color.png'),os.path.join(gtFine_dir, 'cityscapes_semantic_train', file + 'gtFine_color.png'))
            dict['images'].append(meta)
            dict['annotations'].extend(ann_objs)

        print(str(image_i) + '/' + str(len(files)))

    print(str(dir_i + 1) + '/' + str(len(cities)) + ' END PROCCESSING - ' + cities[dir_i])

    # if dir_i >= 1:
    #     break

with open(gtFine_dir+'/cityscapes_semantic_train.json', 'w') as f:
    json.dump(dict, f, indent=2)
#
# for i, file_name in enumerate(file_names):
#     annotation_data = cityscapes2coco(file_name, i)
#     print('Length of total annotations: {}'.format(len(annotation_data["annotations"])))
#
#     # Annotationがないものは省く
#     if len(annotation_data["annotations"]):
#         dict["images"].append(annotation_data["meta"])
#         dict["annotations"].extend(annotation_data["annotations"])
#
# with open('./greenery/train/annotation.json', 'w') as f:
#     json.dump(dict, f, indent=2)
#
# with open('./greenery/detected_items.json', 'w') as f:
#     item_list = sorted(list(set(detected_items)))
#     json.dump({"detected_items": item_list}, f, indent=2)

