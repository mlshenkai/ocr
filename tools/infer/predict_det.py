# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/4/23 17:00
# @Organization: YQN
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import copy
import cv2
import numpy as np
import time

import torch
from ocr.base_ocr import BaseOCR
import tools.utils as utility
from ocr.utils.utility import get_image_file_list, check_and_read_gif
from ocr.data import create_operators, transform
from ocr.postprocess import build_post_process


class TextDetector(BaseOCR):
    def __init__(self, args, **kwargs):
        self.args = args
        self.det_algorithm = args.det_algorithm
        pre_process_list = [{
            'DetResizeForTest': None
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': args.det_limit_side_len
                }
            }
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = args.det_sast_nms_thresh
            self.det_sast_polygon = args.det_sast_polygon
            if self.det_sast_polygon:
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3
        else:
            print("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu

        self.weights_path = args.det_model_path
        self.yaml_path = args.det_yaml_path
        network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path)
        print(network_config)
        super(TextDetector, self).__init__(network_config, **kwargs)

        self.load_weights(self.weights_path)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()
    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def predict(self,img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()
        with torch.no_grad():
            inp = torch.Tensor(img)
            if self.use_gpu:
                inp = inp.cuda()
            outputs = self.net(inp)

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs['f_geo']
            preds['f_score'] = outputs['f_score']
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs['f_border']
            preds['f_score'] = outputs['f_score']
            preds['f_tco'] = outputs['f_tco']
            preds['f_tvo'] = outputs['f_tvo']
        elif self.det_algorithm == 'DB':
            # preds['maps'] = outputs[0]
            preds['maps'] = outputs['maps'].cpu().numpy()
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        if self.det_algorithm == "SAST" and self.det_sast_polygon:
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse

if __name__=="__main__":
    args = utility.parse_args()
    args.det_model_path="../../resources/models/det/ch_ocr_server_v2.0_det_db_infer.pth"
    text_detector=TextDetector(args)
    count = 0
    total_time = 0
    draw_img_save = "./inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    image_file_list = get_image_file_list("../../data/0e9455d58efc4c79b631684a3b731cb8.jpg")
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img=cv2.imread(image_file)
        dt_boxes, elapse = text_detector.predict(img)
        if count > 0:
            total_time += elapse
        count += 1
        print("Predict time of {}: {}".format(image_file, elapse))
        src_im = utility.draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        print("The visualized image saved in {}".format(img_path))


