'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 14:14:18
@FilePath       : /ImageCls.detectron2/imgcls/modeling/backbone/mobilenet.py
@Description    : 
'''


import torch
import torch.nn as nn
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, LastLevelP6P7


__all__ = [
    'build_mnetv1_backbone',
    'build_mnetv2_backbone',
]


def conv_bn_leaky(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw_leaky(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class MobileNetV1(Backbone):
    def __init__(self, cfg, data_channel, width_mult=1.0, out_features=None, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        
        input_channel = 32
        # scale input channel
        input_channel = int(input_channel * width_mult)
        # stem
        current_stride = 2
        name = "stem"
        self.stem = conv_bn_leaky(
            data_channel, input_channel, current_stride, leaky=0.1)

        self._out_feature_strides = {name: current_stride}
        self._out_feature_channels = {name: input_channel}

        # body
        dw_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        self.return_features_indices = [3, 5, 11, 13]
        self.features = nn.ModuleList([])
        # building depthwise conv block
        for c, n, s in dw_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                # the first one applying stride
                if i == 0:
                    self.features.append(conv_dw_leaky(
                        input_channel, output_channel, s))
                else:
                    self.features.append(conv_dw_leaky(
                        input_channel, output_channel, 1))
                # update input channel for next block
                input_channel = output_channel
                # check output this feature map?
                if len(self.features) in self.return_features_indices:
                    name = "mob{}".format(
                        self.return_features_indices.index(len(self.features)) + 2)
                    self._out_feature_channels.update({
                        name: output_channel
                    })
                    current_stride *= 2
                    self._out_feature_strides.update({
                        name: current_stride
                    })

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(input_channel, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def freeze(self, freeze_at):
        if freeze_at > 0:
            # freeze stem
            for p in self.stem.parameters():
                p.requires_grad = False
            if freeze_at > 1:
                # freeze features
                freeze_at = freeze_at - 2
                freeze_layers = self.return_features_indices[freeze_at] if freeze_at < len(
                    self.return_features_indices) else self.return_features_indices[-1]
                for layer_index in range(freeze_layers):
                    for p in self.features[layer_index].parameters():
                        p.requires_grad = False
        return self

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for i, m in enumerate(self.features, 1):
            x = m(x)
            if i in self.return_features_indices:
                name = "mob{}".format(
                    self.return_features_indices.index(i) + 2)
                if name in self._out_features:
                    outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d(inp, hidden_dim, 3, stride, 1,
                       groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride,
                       1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(Backbone):
    def __init__(self, cfg, data_channel, width_mult=1.0, out_features=None, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        
        input_channel = 32
        # scale input channel
        input_channel = int(input_channel * width_mult)
        # stem
        current_stride = 2
        name = "stem"
        self.stem = conv_bn(data_channel, input_channel, current_stride)

        self._out_feature_strides = {name: current_stride}
        self._out_feature_channels = {name: input_channel}

        # body
        block = InvertedResidual
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.return_features_indices = [3, 6, 13, 17]
        self.features = nn.ModuleList([])

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                # the first one applying stride
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t))
                # update input channel for next block
                input_channel = output_channel
                # check output this feature map?
                if len(self.features) in self.return_features_indices:
                    name = "mob{}".format(
                        self.return_features_indices.index(len(self.features)) + 2)
                    self._out_feature_channels.update({
                        name: output_channel
                    })
                    current_stride *= 2
                    self._out_feature_strides.update({
                        name: current_stride
                    })
        
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(input_channel, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def freeze(self, freeze_at):
        if freeze_at > 0:
            # freeze stem
            for p in self.stem.parameters():
                p.requires_grad = False
            if freeze_at > 1:
                # freeze features
                freeze_at = freeze_at - 2
                freeze_layers = self.return_features_indices[freeze_at] if freeze_at < len(
                    self.return_features_indices) else self.return_features_indices[-1]
                for layer_index in range(freeze_layers):
                    for p in self.features[layer_index].parameters():
                        p.requires_grad = False
        return self

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        # res2 -> stride 2**2
        # res3 -> stride 2**3
        # output downsample stride: [4, 8, 16, 32]
        for i, m in enumerate(self.features, 1):
            x = m(x)
            if i in self.return_features_indices:
                name = "mob{}".format(
                    self.return_features_indices.index(i) + 2)
                if name in self._out_features:
                    outputs[name] = x
        
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_mnetv1_backbone(cfg, input_shape: ShapeSpec):
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.MNET.OUT_FEATURES
    width_mult = cfg.MODEL.MNET.WIDTH_MULT
    num_classes = cfg.MODEL.CLSNET.NUM_CLASSES if cfg.MODEL.CLSNET.ENABLE else None
    model = MobileNetV1(cfg, input_shape.channels, width_mult=width_mult,
                        out_features=out_features, num_classes=num_classes).freeze(freeze_at)
    return model


@BACKBONE_REGISTRY.register()
def build_mnetv2_backbone(cfg, input_shape: ShapeSpec):
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.MNET.OUT_FEATURES
    width_mult = cfg.MODEL.MNET.WIDTH_MULT
    num_classes = cfg.MODEL.CLSNET.NUM_CLASSES if cfg.MODEL.CLSNET.ENABLE else None
    model = MobileNetV2(cfg, input_shape.channels, width_mult=width_mult,
                        out_features=out_features, num_classes=num_classes).freeze(freeze_at)
    return model
