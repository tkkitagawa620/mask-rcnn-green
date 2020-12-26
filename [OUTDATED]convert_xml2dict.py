import cv2
import os
import glob
import random
import mimetypes
import json
import base64
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import utils
from detectron2.structures import BoxMode

def xml2json(filename, index):
    """Convert XML to json

        Args:
            xml_path (str): Location of annotated XML file
        Returns:
            dictionary (dict): Annotation data in Dict format

        """
    print("-------------------")
    xml_path = './greenery/train/Annotations_2/' + filename
    image_path = './greenery/train/Images_2/' + filename[:-4] + ".jpg"
    print("XML_PATH: {}".format(xml_path))
    print("IMAGE_PATH: {}".format(image_path))

    record = {}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        obj_list = utils.return_objectList()

        meta = {
            "file_name": utils.ommit_escape(root.find('filename').text),
            "id" : index,
            "width": int(w),
            "height": int(h)
        }

        anns = []
        for ann_index, member in enumerate(root.findall('object')):
            print(member.find('name').text[1:-1])
            detected_items.append(member.find('name').text[1:-1])
            if member.find('name').text[1:-1] in obj_list:
                ann_obj = {
                    "area": utils.calculate_area(member.find('polygon')),
                    "bbox": utils.exrtract_dimensions(member.find('polygon')) ,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [utils.extract_segmentations(member.find('polygon'))],
                    "category_id": 1,
                    "image_id": index,
                    "id": str(index) + "_" + str(ann_index),
                    "iscrowd": 0
                }
                anns.append(ann_obj)

        record["meta"] = meta
        record["annotations"] = anns

    except Exception as e:
        print('xml conversion failed:{}'.format(e))
        return {"annotations": []}
    return record

# ---------------------------------------------------------------------------
# --------------------------------   Script   -------------------------------
# ---------------------------------------------------------------------------

file_names = ["p1010405.xml", "0020.xml"]
file_names = os.listdir("./greenery/train/Annotations_2")
# file_names = random.sample(file_names, 20)
print('create annotations for these 20 files: {}'.format(file_names))

detected_items = []

dict = {
    "images": [],
    "categories": [
        {
            "supercategory": "greenery",
            "id": 1,
            "name": "greenery"
        },
    ],
    "annotations": []
}

for i, file_name in enumerate(file_names):
    # # なぜか処理がうまく回らない
    # if file_name in ["IMG_7896.xml", "IMG_0766.xml"]:
    #     continue

    annotation_data = xml2json(file_name, i)
    print('Length of total annotations: {}'.format(len(annotation_data["annotations"])))

    # Annotationがないものは省く
    if len(annotation_data["annotations"]):
        dict["images"].append(annotation_data["meta"])
        dict["annotations"].extend(annotation_data["annotations"])

with open('./greenery/train/annotation.json', 'w') as f:
    json.dump(dict, f, indent=2)

with open('./greenery/detected_items.json', 'w') as f:
    item_list = sorted(list(set(detected_items)))
    json.dump({"detected_items": item_list}, f, indent=2)

