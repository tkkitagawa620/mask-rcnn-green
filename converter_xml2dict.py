import cv2
import os
import mimetypes
import json
import base64
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode

def ommit_escape(txt):
    return txt[1:-1]

def exrtract_dimensions(polygons):
    xs = []
    ys = []

    for cordinate in polygons.findall('pt'):
        xs.append(int(cordinate.find('x').text))
        ys.append(int(cordinate.find('y').text))

    return [min(xs), min(ys), max(xs), max(ys)]

def extract_segmentations(polygons):
    seg = []
    for cordinate in polygons.findall('pt'):
        seg.append(int(cordinate.find('x').text))
        seg.append(int(cordinate.find('y').text))
    return seg

def xml2json(filename, dir, sub_dir):
    """Convert XML to json

        Args:
            xml_path (str): Location of annotated XML file
        Returns:
            dictionary (dict): Annotation data in Dict format

        """
    print("-------------------")
    xml_path = dir + 'Annotations/' + sub_dir + filename + '.xml'
    image_path = dir + 'Images/' + sub_dir + filename + '.jpg'
    print("xml_path: {}".format(xml_path))
    print("image_path: {}".format(image_path))

    record = {}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        record["filename"] = ommit_escape(root.find('filename').text)
        record["image_id"] = ommit_escape(root.find('filename').text)[:-4]
        record["width"] = int(w)
        record["height"] = int(h)

        objs = []
        for member in root.findall('object'):
            print(member.find('name').text[1:-1])
            if member.find('name').text[1:-1] == 'tree':
                obj = {
                    "bbox": exrtract_dimensions(member.find('polygon')) ,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [extract_segmentations(member.find('polygon'))],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
        record["annotations"] = objs

    except Exception as e:
        print('xml conversion failed:{}'.format(e))
        return {}
    return record


def get_greenery_dicts(dir):
    dataset_dict = []
    sub_dir = 'barcelona_static_street/'
    dirs = os.listdir(dir + 'Annotations/' + sub_dir)
    for file in dirs:
        if file[-3:] == "xml":
            filename = file[:-4]
            dataset_dict.append(xml2json(filename, dir, sub_dir))
    return dataset_dict

# -------------------------

# dataset_dict = []
#
# dir = 'raw_datasets/'
# sub_dir = 'barcelona_static_street/'
# dirs = os.listdir(dir + 'Annotations/' + sub_dir)
# for file in dirs:
#     if file[-3:] == "xml":
#         filename = file[:-4]
#         dataset_dict.append(xml2json(filename, dir, sub_dir))
# filename = file[:-4]
# print(filename)
# dataset_dict.append(xml2json(filename, dir, sub_dir))

# print(dataset_dict)
#
# with open('test.json', 'w') as f:
#     json.dump(dataset_dict, f)
