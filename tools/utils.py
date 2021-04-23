# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/4/23 15:24
# @Organization: YQN
import argparse
import os
from pathlib import Path

def parse_args():
    def str2bool(v: str):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_path", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_path", type=str)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)

    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)
    parser.add_argument("--limited_max_width", type=int, default=1280)
    parser.add_argument("--limited_min_width", type=int, default=16)

    parser.add_argument(
        "--vis_font_path", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'doc/fonts/simfang.ttf'))
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'ocr/utils/dict/ch_keys.txt'))

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_path", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)

    # params .yaml
    parser.add_argument("--det_yaml_path", type=str, default=None)
    parser.add_argument("--rec_yaml_path", type=str, default=None)
    parser.add_argument("--cls_yaml_path", type=str, default=None)

    return parser.parse_args()


def get_default_config(args):
    return vars(args)


def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))
    return res['Architecture']

def AnalysisConfig(weights_path, yaml_path=None):
    if not os.path.exists(os.path.abspath(weights_path)):
        raise FileNotFoundError('{} is not found.'.format(weights_path))

    if yaml_path is not None:
        return read_network_config_from_yaml(yaml_path)

    weights_basename = os.path.basename(weights_path)
    weights_name = weights_basename.lower()


    if weights_name == 'ch_ptocr_server_v2.0_det_infer.pth':
        network_config = {'model_type':'det',
                          'algorithm':'DB',
                          'Transform':None,
                          'Backbone':{'name':'ResNet', 'layers':18, 'disable_se':True},
                          'Neck':{'name':'DBFPN', 'out_channels':256},
                          'Head':{'name':'DBHead', 'k':50}}

    elif weights_name == 'ch_ptocr_server_v2.0_rec_infer.pth':
        network_config = {'model_type':'rec',
                          'algorithm':'CRNN',
                          'Transform':None,
                          'Backbone':{'name':'ResNet', 'layers':34},
                          'Neck':{'name':'SequenceEncoder', 'hidden_size':256, 'encoder_type':'rnn'},
                          'Head':{'name':'CTCHead', 'fc_decay': 4e-05}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'DBFPN', 'out_channels': 96},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_rec_infer.pth':
        network_config = {'model_type':'rec',
                          'algorithm':'CRNN',
                          'Transform':None,
                          'Backbone':{'model_name':'small', 'name':'MobileNetV3', 'scale':0.5, 'small_stride':[1,2,2,2]},
                          'Neck':{'name':'SequenceEncoder', 'hidden_size':48, 'encoder_type':'rnn'},
                          'Head':{'name':'CTCHead', 'fc_decay': 4e-05}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_cls_infer.pth':
        network_config = {'model_type':'cls',
                          'algorithm':'CLS',
                          'Transform':None,
                          'Backbone':{'name':'MobileNetV3', 'model_name':'small', 'scale':0.35},
                          'Neck':None,
                          'Head':{'name':'ClsHead', 'class_dim':2}}

    else:
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5,
                                       'small_stride': [1, 2, 2, 2]},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}
        # raise NotImplementedError

    return network_config
