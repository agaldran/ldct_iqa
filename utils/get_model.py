import sys, os
import numpy as np
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.convnext import LayerNorm2d
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.models import swin_s, Swin_S_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from . import bit_models

class MultiHeadModel(nn.Module):
    def __init__(self, model_name, num_classes, num_heads, pre_softmax=True, weights=None):
        super(MultiHeadModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.pre_softmax = pre_softmax

        if model_name=='resnet18':
            self.model = resnet18(weights=weights)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_name=='resnet34':
            self.model = resnet34(weights=weights)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_name=='resnet50':
            self.model = resnet50(weights=weights)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_name == 'convnext':
            self.model = convnext_tiny(weights=weights)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Identity()
        elif model_name == 'mobilenet_v2':
            self.model = mobilenet_v2(weights=weights)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Identity()
        elif model_name == 'eff_b1':
            self.model = efficientnet_b1(weights=weights)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Identity()
        elif model_name == 'eff_b2':
            self.model = efficientnet_b2(weights=weights)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Identity()
        elif model_name == 'eff_v2':
            self.model = efficientnet_v2_s(weights=weights)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Identity()
        elif model_name == 'eff_v2_M':
            self.model = efficientnet_v2_m(weights=weights)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Identity()
        elif model_name == 'swin':
            self.model = swin_t(weights=weights)
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Identity()
        elif model_name == 'swinV2':
            self.model = swin_v2_t(weights=weights)
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Identity()
        elif model_name == 'swinS':
            self.model = swin_s(weights=weights)
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Identity()
        elif model_name == 'bit_resnext50_1':
            self.model = bit_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
            if weights:
                if not os.path.isfile('models/BiT-M-R50x1.npz'):
                    print('downloading bit_resnext50_1 weights:')
                    os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz -P models/')
                self.model.load_from(np.load('models/BiT-M-R50x1.npz'))
            self.model.head = nn.Identity()
            num_ftrs = 0

        else: sys.exit('model not defined')

        def get_head(model_name, num_ftrs, num_classes):
            if model_name == 'convnext':
                return nn.Sequential(LayerNorm2d([num_ftrs, ], eps=1e-06, elementwise_affine=True),
                                     nn.Flatten(start_dim=1, end_dim=-1),
                                     nn.Linear(in_features=num_ftrs, out_features=num_classes))
            elif model_name == 'bit_resnext50_1':
                from collections import OrderedDict
                head_size = num_classes
                wf = 1
                return nn.Sequential(OrderedDict([('gn', nn.GroupNorm(32, 2048*wf)),
                                                  ('relu', nn.ReLU(inplace=True)),
                                                  ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
                                                  ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True))])
                                     )
            else: return nn.Linear(num_ftrs, num_classes)

        self.heads = nn.ModuleList([get_head(model_name, num_ftrs, num_classes) for _ in range(self.num_heads)])

        # we are doing imagenet weights here; maybe it would have been a good idea to initialize the heads but meh now.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, 0.01)
        #         m.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        head_predictions = [self.heads[i](x).squeeze() for i in range(self.num_heads)]
        out = torch.stack(head_predictions, dim=-2)  # batch_size x num_heads x num_classes

        if self.model.training:
            return out  # batch_size x num_heads x num_classes
        else:
            if self.pre_softmax:
                # mean of logits over heads.softmax(-1), batch_size x_numclasses, this normalizes each head and keeps shape
                s = out.softmax(dim=-1)
                # note to future self:
                # yes it does, don't freak out; you thought at first that it would normalize without distinguishing one
                # head from another, but then you checked and it is alright. See, run this:
                # x = torch.ones(8,2,4)
                # s = x.softmax(-1)
                # print(s[0,0])
                # print(s[0,1])
                # it gives 0.25s, not 0.125s, thankfully. Move on.
                out = torch.mean(s, dim=-2)
                return out
            else:
                # mean over heads, return just the logits and the user takes care of softmaxing, req. for Temp. Scaling
                return torch.mean(out, dim=-2)


def get_arch(model_name, n_classes):
    if model_name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnet34':
        weights = ResNet34_Weights.IMAGENET1K_V1
        model = resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif model_name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'eff_b1':
        weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        model = efficientnet_b1(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_classes)
    elif model_name == 'eff_b2':
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        model = efficientnet_b2(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'swin':
        weights = Swin_T_Weights.IMAGENET1K_V1
        model = swin_t(weights=weights)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    else:
        sys.exit('{} is not a valid model_name, check utils.get_model.py'.format(model_name))

    setattr(model, 'n_classes', n_classes)
    return model



