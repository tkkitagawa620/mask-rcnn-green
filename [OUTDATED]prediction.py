import os
import random
import cv2
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

register_coco_instances("greenery", {}, "./greenery/train/annotation.json", "./greenery/train/Images_2")
greenery_meta_data = MetadataCatalog.get("greenery")
dataset_dicts = DatasetCatalog.get("greenery")

# make prediction
cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (300)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set the testing threshold for this model
cfg.DATASETS.TEST = ("greenery", )
cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)

im = cv2.imread('./street_img_2.jpg')
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
   metadata=greenery_meta_data,
   scale=0.8,
   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('prediction', v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()