import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, align_resnet=False):
        super(ResidualBlock, self).__init__()

        self.align_resnet = align_resnet # New paremeter for task 5
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))

        '''
        Task 5: Align the residual unit to the original ResNet paper.
        Original CCMR(align_resnet = False): conv1 -> norm1 -> relu -> conv2 -> norm2 -> relu
        Modification(align_resnet = True): remove the last relu.
        '''
        if self.align_resnet:
            y = self.norm2(self.conv2(y))
        else:
            y = self.relu(self.norm2(self.conv2(y)))    
        
        if self.downsample is not None:
            x = self.downsample(x)
        if x.shape[1] != y.shape[1]: # for the uplayers.
            return y
       
        return self.relu(x+y)


class BasicEncoder_resconv(nn.Module):
    def __init__(self, output_dim=128, norm_fn='group', model_type="CCMR+", align_resnet = False, remove_consolidation=False, one_scale_16 = False, one_scale_8 = False, one_scale_8_conv=False):
        super(BasicEncoder_resconv, self).__init__()
        self.align_resnet = align_resnet
        self.remove_consolidation = remove_consolidation
        self.one_scale_16 = one_scale_16
        self.one_scale_8 = one_scale_8
        self.one_scale_8_conv = one_scale_8_conv
        self.norm_fn = norm_fn
        self.model_type = model_type
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        #Top-down feature extraction

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(160, stride=2)
        self.conv2 = nn.Conv2d(160, 160, kernel_size=1)

        # quick experiment for adding back activation free convolution layer for one_scale_8_conv_it. 22. Oct
        self.one_scale_out_conv = nn.Conv2d(128, 128, kernel_size=1)
        '''
        Task 4: Remove feature consolidation and only use a top-down extractor.
        remove_consolidation = True -> only use a top-down extractor without consolidation.
        remove_consolidation = False -> original CCMR process.
        '''
        if self.remove_consolidation:
            pass
        else:
            #Bottom-up feature consolidation
            
            self.in_planes = 160 + 128
            self.up_layer2 = self._make_layer(128, stride=1)
            self.after_up_layer2_conv = nn.Conv2d(128, 128, kernel_size=1) 
            self.in_planes = 128 + 96
            self.up_layer1 = self._make_layer(96, stride=1)
            self.after_up_layer1_conv = nn.Conv2d(96, 96, kernel_size=1)

            if self.model_type.casefold() == "ccmr+":
                self.in_planes = 96 + 64
                self.up_layer0 = self._make_layer(64, stride=1)
                self.after_up_layer0_conv = nn.Conv2d(64, 64, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride,self.align_resnet)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, 1, self.align_resnet)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

       
        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x        
        
        # Task 2 one scale: (one_scale_16 has 1/16 resolution; one_scale_8 and one_scale_8_conv have 1/8 resolution)
        if self.one_scale_16:
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            return [enc_out4]
        elif self.one_scale_8 or self.one_scale_8_conv:
            # quick experiment for adding back activation free convolution layer for one_scale_8_conv_it. 22. Oct
            enc_out3 = self.one_scale_out_conv(enc_out3)
            enc_out3 = torch.split(enc_out3, [batch_dim, batch_dim], dim=0)
            return [enc_out3]

        # Task 4 modification:
        if self.remove_consolidation:
            if self.model_type.casefold() == "ccmr+":
                enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
                enc_out3 = torch.split(enc_out3, [batch_dim, batch_dim], dim=0)
                enc_out2 = torch.split(enc_out2, [batch_dim, batch_dim], dim=0)
                enc_out1= torch.split(enc_out1, [batch_dim, batch_dim], dim=0)
                return [enc_out4, enc_out3, enc_out2, enc_out1]
            elif self.model_type.casefold() == "ccmr":
                enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
                enc_out3 = torch.split(enc_out3, [batch_dim, batch_dim], dim=0)
                enc_out2 = torch.split(enc_out2, [batch_dim, batch_dim], dim=0)
                return [enc_out4, enc_out3, enc_out2]
        else:    
            #uplayer2:
            cur_h, cur_w = list(enc_out3.size())[-2:]
            enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
            up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
            up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))

            #uplayer1:
            cur_h, cur_w = list(enc_out2.size())[-2:]
            up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
            up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
            up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))

            if self.model_type.casefold() == "ccmr+":
                #uplayer0:
                cur_h, cur_w = list(enc_out1.size())[-2:]
                up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
                up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
                up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))
                enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
                up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
                up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)
                up0_out = torch.split(up0_out, [batch_dim, batch_dim], dim=0)
                return [enc_out4, up2_out, up1_out, up0_out]    
            elif self.model_type.casefold() == "ccmr":
                enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
                up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
                up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)
                return [enc_out4, up2_out, up1_out]    

        
class Basic_Context_Encoder_resconv(nn.Module):
    def __init__(self, output_dim=128, norm_fn='group', model_type="CCMR+", align_resnet=False, remove_consolidation=False, one_scale_16=False, one_scale_8 = False, one_scale_8_conv=False):
        super(Basic_Context_Encoder_resconv, self).__init__()
        self.align_resnet = align_resnet
        self.remove_consolidation = remove_consolidation
        self.one_scale_16 = one_scale_16
        self.one_scale_8 = one_scale_8
        self.one_scale_8_conv = one_scale_8_conv
        self.norm_fn = norm_fn
        self.model_type = model_type
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        #Top-down feature extraction

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(160, stride=2)
        self.conv2 = nn.Conv2d(160, output_dim, kernel_size=1)
        
        if self.one_scale_8 or self.one_scale_8_conv:
            self.projection3_single = nn.Conv2d(128, output_dim, kernel_size = 1)

        # Task 4 modification:
        # Without consolidation, we still need a projection to change the number of channels to output_dim.
        elif self.remove_consolidation:
            self.projection3 = nn.Conv2d(128, output_dim, kernel_size = 1)
            self.projection2 = nn.Conv2d(96, output_dim, kernel_size = 1)
            if self.model_type.casefold() == "ccmr+":
                self.projection1 = nn.Conv2d(64, output_dim, kernel_size = 1)
        else:
            #Bottom-up feature consolidation
            self.in_planes = 256 + 128
            self.up_layer2 = self._make_layer(128, stride=1)
            self.after_up_layer2_conv = nn.Conv2d(128, output_dim, kernel_size=1)
            self.in_planes = 256 + 96
            self.up_layer1 = self._make_layer(96, stride=1)
            self.after_up_layer1_conv = nn.Conv2d(96, output_dim, kernel_size=1) 
            if self.model_type.casefold() == "ccmr+":
                self.in_planes = 256 + 64
                self.up_layer0 = self._make_layer(64, stride=1)
                self.after_up_layer0_conv = nn.Conv2d(64, output_dim, kernel_size=1) 
       

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride, self.align_resnet)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, 1, self.align_resnet)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):
      
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x
        
        # Task 2 one scale:
        if self.one_scale_16:
            return [enc_out4]
        elif self.one_scale_8 or self.one_scale_8_conv:
            enc_out3_projected = self.projection3_single(enc_out3)
            return [enc_out3_projected]

        # Task 4 modification:
        if self.remove_consolidation:
            enc_out3_projected = self.projection3(enc_out3)
            enc_out2_projected = self.projection2(enc_out2)
            if self.model_type.casefold() == "ccmr+":
                enc_out1_projected = self.projection1(enc_out1)
                return [enc_out4, enc_out3_projected, enc_out2_projected, enc_out1_projected]
            elif self.model_type.casefold() == "ccmr":
                return [enc_out4, enc_out3_projected, enc_out2_projected]
        else:
            #uplayer2:
            cur_h, cur_w = list(enc_out3.size())[-2:]
            enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
            up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
            up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))
            #uplayer1:
            cur_h, cur_w = list(enc_out2.size())[-2:]
            up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
            up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
            up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))
            if self.model_type.casefold() == "ccmr+":
                #uplayer0:
                cur_h, cur_w = list(enc_out1.size())[-2:]
                up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
                up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
                up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))
                return [enc_out4, up2_out, up1_out, up0_out]  
            
            elif self.model_type.casefold() == "ccmr":
                return [enc_out4, up2_out, up1_out] 
