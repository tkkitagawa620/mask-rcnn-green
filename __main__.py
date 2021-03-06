import detectron2
import os
import cv2
import random
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger

setup_logger()

# register dataset
register_coco_instances("cityscapes", {}, "./datasets/cityscapes/gtFine/cityscapes_semantic_train.json", "./datasets/cityscapes/leftImg8bit/semantic_train")
cityscapes_meta_data = MetadataCatalog.get("cityscapes")
dataset_dicts = DatasetCatalog.get("cityscapes")

# validate dataset
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:,:,::-1], metadata=cityscapes_meta_data, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1] )
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# configure detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("cityscapes",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (100)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE='cpu'

# # train
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# # trainer.train()
#
# make prediction
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("cityscapes",)
predictor = DefaultPredictor(cfg)

sample = os.path.join('datasets/cityscapes/leftImg8bit/test/berlin', random.choice(os.listdir('datasets/cityscapes/leftImg8bit/test/berlin')))
sample = os.path.join('datasets/cityscapes/leftImg8bit/test/berlin/IMG_1660.jpg')
im = cv2.imread(sample)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
   metadata=cityscapes_meta_data,
   scale=0.8,
   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('predicted image', v.get_image()[:,:,::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('output/prediction/original.png', im)
# cv2.imwrite('output/prediction/predicted.png', v.get_image()[:,:,::-1])