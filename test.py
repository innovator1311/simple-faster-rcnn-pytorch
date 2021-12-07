import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('/content/fasterrcnn_12051807_1.0000000000000002')
opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model

img = read_image('/content/trial/JPEGImages/VO_TAN_HUNG.MR.BENH_VIEN_C_DA_NANG_KHOP_GOI.0009.0014.2021.08.17.12.30.09.692036.293359582.jpg')
img = t.from_numpy(img)[None]
_bboxes, _labels, _scores, rpn_locs, rpn_scores = trainer.faster_rcnn.predict(img,visualize=True)

vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))

