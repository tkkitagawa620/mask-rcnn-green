import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import torch
# print(torch.cuda.is_available())

# 画像の読み込み
im = cv2.imread('./sample_image.jpeg')

# Detectron2の設定とモデル固有の設定み読み込む
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE='cpu'

# 閾値の設定
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

# トレーニング済みのファイルを読み込み
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# 推論の実行
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# # 結果の可視化
v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('speculated image', v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()