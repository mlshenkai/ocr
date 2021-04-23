# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/4/23 17:01
# @Organization: YQN

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ocr.model.base_model import BaseModel


class BaseOCR:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()

    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)

    def get_out_channels(self, weights):
        out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)

    def load_weights(self, weight_path):
        self.net.load_state_dict(torch.load(weight_path))
        print("model is loaded...")

    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer
