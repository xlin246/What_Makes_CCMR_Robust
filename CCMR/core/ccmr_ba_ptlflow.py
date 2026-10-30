from pathlib import Path
import sys
from typing import Sequence

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ptlflow-main"))
sys.path.insert(0, str(Path(__file__).parent))
import ptlflow
from ptlflow.models.ccmr.corr import get_corr_block
from ptlflow.models.ccmr.utils import coords_grid, upflow2
from ptlflow.utils.registry import register_model
from ptlflow.utils.utils import forward_interpolate_batch
from ptlflow.models.base_model.base_model import BaseModel


from .update import BasicUpdateBlock
from .extractor import BasicEncoder_resconv, Basic_Context_Encoder_resconv
#from ptlflow.models.ccmr.corr import get_corr_block
#from .utils.utils import coords_grid, upflow2
from .xcit import XCiT

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None

# loss function from RAFT
class SequenceLoss(nn.Module):
    def __init__(self, gamma: float, max_flow: float):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow

    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""

        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        return flow_loss

def downflow(flow, mode="bilinear", factor=0.125):
    old_size = (flow.shape[2], flow.shape[3])
    new_size = (int(factor * flow.shape[2]), int(factor * flow.shape[3]))
    u_scale = new_size[1] / old_size[1]
    v_scale = new_size[0] / old_size[0]
    resized_flow = F.interpolate(
        flow, size=new_size, mode=mode, align_corners=True
    )  # b 2 h w
    resized_flow_split = torch.split(resized_flow, 1, dim=1)
    rescaled_flow = torch.cat(
        [u_scale * resized_flow_split[0], v_scale * resized_flow_split[1]], dim=1
    )

    return rescaled_flow


class CCMR_BA(BaseModel):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ccmr-kitti-612444b9.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ccmr-sintel-e1760f37.ckpt",
    }

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: Sequence[int] = [4, 6, 8],
        alternate_corr: bool = True,
        lookup_pyramid_levels: int = 2,
        lookup_radius: int = 4,
        model_type: str = "CCMR",
        cnet_norm: str = "group",
        fnet_norm: str = "group",
        num_scales: int = 3,
        remove_motion: bool = False,
        align_resnet: bool = False,
        remove_consolidation:bool = False,
        one_scale_16:bool = False,
        disable_shared_parameter: bool = False,
        one_scale_8:bool = False,
        one_scale_8_conv:bool = False,
        **kwargs,
    ) -> None:
        loss_fn = SequenceLoss(gamma=0.8, max_flow=400)
        super().__init__(output_stride=32, loss_fn=loss_fn, **kwargs)
        
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.iters = iters
        self.alternate_corr = alternate_corr
        self.lookup_pyramid_levels = lookup_pyramid_levels
        self.lookup_radius = lookup_radius
        self.model_type = model_type
        self.cnet_norm = cnet_norm
        self.fnet_norm = fnet_norm
        self.num_scales = num_scales
        self.remove_motion = remove_motion
        self.align_resnet = align_resnet
        self.remove_consolidation = remove_consolidation
        self.one_scale_16 = one_scale_16
        self.disable_shared_parameter = disable_shared_parameter
        self.one_scale_8 = one_scale_8
        self.one_scale_8_conv = one_scale_8_conv
        
        if self.one_scale_16 or self.one_scale_8 or self.one_scale_8_conv:
            self.num_scales = 1
        
        if (self.one_scale_16 or self.one_scale_8 or self.one_scale_8_conv) and len(self.iters) > 1:
            self.iters = [self.iters[0]] 

        # Initiating the improved multi-scale feature encoders:
        self.fnet = BasicEncoder_resconv(
            output_dim=256, norm_fn=self.fnet_norm, model_type=self.model_type, align_resnet=self.align_resnet, remove_consolidation = self.remove_consolidation, one_scale_16 = self.one_scale_16, one_scale_8 = self.one_scale_8, one_scale_8_conv = self.one_scale_8_conv
        )
        self.cnet = Basic_Context_Encoder_resconv(
            output_dim=256, norm_fn=self.cnet_norm, model_type=self.model_type, align_resnet=self.align_resnet, remove_consolidation = self.remove_consolidation, one_scale_16 = self.one_scale_16, one_scale_8 = self.one_scale_8, one_scale_8_conv = self.one_scale_8_conv
        )
        args = {"remove_motion": self.remove_motion}

        if self.one_scale_8_conv:
            convex_scale = 8
        else:
            convex_scale = 2

        if self.disable_shared_parameter:
            self.update_block = nn.ModuleList()
            for _ in range(self.num_scales):
                block = BasicUpdateBlock(
                    args,
                    correlation_depth=(2 * self.lookup_radius + 1) ** 2
                    * self.lookup_pyramid_levels,
                    hidden_dim=128,
                    scale=convex_scale,
                    num_heads=8,
                    depth=1,
                    mlp_ratio=1,
                    num_scales=self.num_scales,
                    remove_motion=self.remove_motion
                )
                self.update_block.append(block)
        else:
            self.update_block = BasicUpdateBlock(
                    args,
                    correlation_depth=(2 * self.lookup_radius + 1) ** 2
                    * self.lookup_pyramid_levels,
                    hidden_dim=128,
                    scale=convex_scale,
                    num_heads=8,
                    depth=1,
                    mlp_ratio=1,
                    num_scales=self.num_scales,
                    remove_motion=self.remove_motion
                )

        # Initiate global context grouping modules
        if not self.remove_motion:
            self.xcit = nn.ModuleList(
                [XCiT(embed_dim=128, depth=1, num_heads=8, mlp_ratio=1, separate=False)]
            )
            for i in range(self.num_scales - 1):
                self.xcit.extend(
                    [XCiT(embed_dim=128, depth=1, num_heads=8, mlp_ratio=1, separate=False)]
                )

        if self.alternate_corr and alt_cuda_corr is None:
            logger.warning(
                "!!! alt_cuda_corr is not compiled! The slower IterativeCorrBlock will be used instead !!!"
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow16(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 16, W // 16, dtype=img.dtype, device=img.device)
        coords1 = coords_grid(N, H // 16, W // 16, dtype=img.dtype, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def initialize_flow8(self,img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, dtype=img.dtype, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, dtype=img.dtype, device=img.device)
        return coords0, coords1

    def get_grid(self, img, scale):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(
            N, H // scale, W // scale, dtype=img.dtype, device=img.device
        )
        return coords0

    def upsample_flow(self, flow, mask, scale=8):
        """Upsample flow field [H/scale, W/scale, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale * H, scale * W)

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        fnet_pyramid = self.fnet([image1, image2])
        cnet_pyramid = self.cnet(image1)

        if self.one_scale_8 or self.one_scale_8_conv:
            coords0, coords1 = self.initialize_flow8(image1)
        else:
            coords0, coords1 = self.initialize_flow16(image1)

        flow_predictions = []

        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            forward_flow = forward_interpolate_batch(inputs["prev_preds"]["flow_small"])
            coords1 = coords1 + forward_flow

        assert len(fnet_pyramid) == len(
            cnet_pyramid
        ), "fnet and cnet pyramid should have the same length."
        assert len(fnet_pyramid) == len(
            self.iters
        ), "pyramid levels and the length of GRU iteration lists should be the same."
        
        # task 2: one scale variant only needs convex upsampler
        if not (self.one_scale_16 or self.one_scale_8 or self.one_scale_8_conv):
            upsampling_offset = (
                self.num_scales - 1 if self.num_scales == 4 else self.num_scales
            )
            
        for index, (fmap1, fmap2) in enumerate(fnet_pyramid):
            corr_fn = get_corr_block(
                fmap1=fmap1,
                fmap2=fmap2,
                radius=self.lookup_radius,
                num_levels=self.lookup_pyramid_levels,
                alternate_corr=self.alternate_corr,
            )

            net, inp = torch.split(cnet_pyramid[index], [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            global_context = self.xcit[index](inp) if not self.remove_motion else None

            for itr in range(self.iters[index]):
                coords1 = coords1.detach()
                if index >= 1 and itr == 0:
                    flow = self.upsample_flow(
                        coords1 - coords0, up_mask, scale=2
                    )  # bug of MS-RAFT's flow upsampling fixed here.

                    coords0 = self.get_grid(image1, scale=16 / (2**index))

                    coords1 = coords0 + flow

                corr = corr_fn(coords1)
                flow = coords1 - coords0
                if self.disable_shared_parameter:
                    net, up_mask, delta_flow = self.update_block[index](
                    net, inp, corr, flow, global_context, level_index=index
                    )
                else:
                    net, up_mask, delta_flow = self.update_block(
                    net, inp, corr, flow, global_context, level_index=index
                    )

                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow
                # upsample predictions
                if self.one_scale_16:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask, scale=2)
                    for i in range(3):
                        flow_up = upflow2(flow_up)
                elif self.one_scale_8:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask, scale=2)
                    for i in range(2):
                        flow_up = upflow2(flow_up)
                elif self.one_scale_8_conv:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask, scale = 8)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask, scale=2)
                    for i in range(upsampling_offset - index):
                        flow_up = upflow2(flow_up)
                
                flow_up = self.postprocess_predictions(
                    flow_up, image_resizer, is_flow=True
                )
                flow_predictions.append(flow_up)

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        else:
            outputs = {
                "flows": flow_up[:, None],
                "flow_small": downflow(flow_up, factor=0.0625),
            }

        return outputs


class CCMR_BA_Plus(CCMR_BA):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ccmr+-kitti-c289d5e6.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ccmr+-sintel-055b44ec.ckpt",
    }

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: Sequence[int] = [8, 10, 10, 10],
        alternate_corr: bool = True,
        lookup_pyramid_levels: int = 2,
        lookup_radius: int = 4,
        model_type: str = "CCMR+",
        cnet_norm: str = "group",
        fnet_norm: str = "group",
        num_scales: int = 4,
        remove_motion: bool = False,
        align_resnet: bool = False,
        remove_consolidation: bool = False,
        one_scale_16: bool = False,
        disable_shared_parameter: bool = False,
        one_scale_8:bool = False,
        one_scale_8_conv:bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            corr_levels,
            corr_radius,
            iters,
            alternate_corr,
            lookup_pyramid_levels,
            lookup_radius,
            model_type,
            cnet_norm,
            fnet_norm,
            num_scales,
            remove_motion,
            align_resnet,
            remove_consolidation,
            one_scale_16,
            disable_shared_parameter,
            one_scale_8,
            one_scale_8_conv,
            **kwargs,
        )


@register_model
class ccmr_ba(CCMR_BA):
    pass


@register_model
class ccmr_ba_p(CCMR_BA_Plus):
    pass
