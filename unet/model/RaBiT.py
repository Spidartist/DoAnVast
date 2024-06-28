import torch.nn as nn
import torch.nn.functional as F
from .include.conv_layer import Conv
from .utils import BiRAFPN
from .include.axial_atten import AA_kernel
from .include.context_module import CFPModule

class RaBiTSegmentor(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 compound_coef=4,
                 numrepeat = 4,
                 in_channels=(192, 384, 768),
                 ):
        super(RaBiTSegmentor, self).__init__()
        self.in_channels = in_channels
        
        self.backbone = backbone
        
        self.neck = neck
            
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.numrepeat = numrepeat + 1
        self.compound_coef = compound_coef
        self.conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],#448
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }
        
        self.head_seg = self.build_head_segment_rabit()

    def build_head_segment_rabit(self):
        head_segment = nn.ModuleDict()
        
        head_segment["bifpn"] = nn.Sequential(
                                  *[BiRAFPN(self.fpn_num_filters[self.compound_coef],
                                          self.conv_channel_coef[self.compound_coef],
                                          True if _ == 0 else False,
                                          attention=True ,
                                          use_p8=self.compound_coef > 7)
                                    for _ in range(self.numrepeat)]
                                )

        head_segment["conv1"] = Conv(768,self.conv_channel_coef[self.compound_coef][0],1,1,padding=0,bn_acti=True)
        head_segment["conv2"] = Conv(768,self.conv_channel_coef[self.compound_coef][1],1,1,padding=0,bn_acti=True)
        head_segment["conv3"] = Conv(768,self.conv_channel_coef[self.compound_coef][2],1,1,padding=0,bn_acti=True)
        head_segment["head1"] = Conv(self.fpn_num_filters[self.compound_coef],1,1,1,padding=0,bn_acti=False)
        head_segment["head2"] = Conv(self.fpn_num_filters[self.compound_coef],1,1,1,padding=0,bn_acti=False)
        head_segment["head3"] = Conv(self.fpn_num_filters[self.compound_coef],1,1,1,padding=0,bn_acti=False)

        return head_segment

    def forward_segment_rabit(self, head, x):
        x1 = x[0] # 1/4
        x2 = x[1] # 1/8
        x3 = x[2] # 1/16
        x4 = x[3] # 1/32
        
        x2 = head["conv1"](x2)
        x3 = head["conv2"](x3)
        x4 = head["conv3"](x4)
        p3, p4, p5, p6, p7 = head["bifpn"]([x2,x3,x4])
        p3 = head["head3"](p3)
        p4 = head["head2"](p4)
        p5 = head["head1"](p5)
        
        lateral_map_2 = F.interpolate(p5,scale_factor=32,mode='bilinear')
        lateral_map_5 = F.interpolate(p3,scale_factor=8,mode='bilinear') 
        lateral_map_3 = F.interpolate(p4,scale_factor=16,mode='bilinear') 
        lateral_map_1 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1

    def forward(self, inputs):
        x = self.backbone(inputs, return_orders=True, return_features=True)
        x = self.neck(x)
        map = self.forward_segment_rabit(self.head_seg, x)
        
        return {"map" : map}