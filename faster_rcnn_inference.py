from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt

im = cv2.imread("/home/juravlik/Desktop/original_3.png")

cfg = get_cfg()
cfg.merge_from_file("configs/faster_rcnn.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "/home/juravlik/Downloads/faster_rcnn.pth"

predictor = DefaultPredictor(cfg)

outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

for box, score in zip(outputs["instances"].pred_boxes.to('cpu'), outputs["instances"].scores.to('cpu')):
    v.draw_box(box, edge_color='blue')
    # v.draw_text(
    #     'airplane ' + str(np.around(score.numpy(), 5) * 100) + '%',
    #     tuple(box[:2].numpy()),
    #     color='green',
    #     font_size=12,
    # )

v = v.get_output()

plt.imshow(v.get_image())
plt.show()
