import os
import json
import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
# from torchvision.models.video import s3d, S3D_Weights
from lib.utils.tensor_utils import pad_sequences_1d, pad_sequences_2d


class ViTBackbone(nn.Module):

    def __init__(self, video_feature_extractor, sketch_feature_extractor,
                 video_backbone, sketch_backbone,
                 norm_sketch_feats=True, use_sketch_cls_token=True,
                 norm_vid_feats=True, use_vid_cls_token=True):

        super(ViTBackbone, self).__init__()
        self.video_feature_extractor = video_feature_extractor
        self.sketch_feature_extractor = sketch_feature_extractor
        self.video_backbone = video_backbone
        self.sketch_backbone = sketch_backbone
        self.norm_sketch_feats = norm_sketch_feats
        self.use_sketch_cls_token = use_sketch_cls_token
        self.norm_vid_feats = norm_vid_feats
        self.use_vid_cls_token = use_vid_cls_token

    def forward(self, sketch_batch, video_batch):
        src_sketch = []
        for sketch in sketch_batch:
            input_sketch = self.sketch_feature_extractor(images=[sketch], return_tensors="pt")
            output_sketch = self.sketch_backbone(pixel_values=input_sketch['pixel_values'].to(device), output_hidden_states=True)
            if self.norm_sketch_feats:
                output_sketch = output_sketch.last_hidden_state.squeeze()
            else:
                output_sketch = output_sketch.hidden_states[-1].squeeze()
            if self.use_sketch_cls_token:
                output_sketch = output_sketch[0].contiguous()  # (D, )
            else:
                output_sketch = output_sketch[1:].mean(0)  # (D, )
            src_sketch.append(output_sketch)

        src_sketch = torch.stack(src_sketch).unsqueeze(1)  # (N, 1, D)

        src_video = []
        for video in video_batch:
            output_video = []
            for frame in video:
                input_frame = self.video_feature_extractor(images=[frame], return_tensors="pt")
                output_frame = self.video_backbone(pixel_values=input_frame['pixel_values'].to(device), output_hidden_states=True)
                if self.norm_vid_feats:
                    output_video.append(output_frame.last_hidden_state.squeeze())
                else:
                    output_video.append(output_frame.hidden_states[-1].squeeze())
            output_video = torch.stack(output_video)
            if self.use_vid_cls_token:
                output_video = output_video[:,0,:].contiguous()  # (T, D)
            else:
                output_video = output_video[:,1:,:].mean(1)  # (T, D)
            src_video.append(output_video)

        return src_sketch, src_video


class ResNetBackbone(nn.Module):

    def __init__(self, video_backbone, sketch_backbone):
        super(ResNetBackbone, self).__init__()
        self.video_backbone = video_backbone
        self.sketch_backbone = sketch_backbone

    def forward(self, sketch_batch, video_batch):
        '''
        sketch_batch: [N, 1, C, H, W]
        video_batch: [N, T, C, H, W]
        '''
        sketch_batch = sketch_batch.squeeze(1)  # (N, C0, H0, W0)
        src_sketch = self.sketch_backbone(sketch_batch).squeeze()  # (N, C)
        src_sketch = src_sketch.unsqueeze(1)  # (N, 1, C)

        N, T, C0, H0, W0 = video_batch.shape
        video_batch = video_batch.flatten(0, 1)  # (N*T, C0, H0, W0)
        src_video = self.video_backbone(video_batch)  # (N*T, C, H, W)
        src_video = src_video.reshape(N, T, *src_video.shape[1:])  # (N, T, C, H, W)
        src_video = src_video.transpose(1, 2)  # (N, C, T, H, W)
        src_video = src_video.flatten(2, -1)  # (N, C, T*H*W)
        src_video = src_video.transpose(1, 2)  # (N, T*H*W, C)

        return src_sketch, src_video


# class S3DBackbone(nn.Module):

#     def __init__(self, video_backbone, sketch_backbone):
#         super(S3DBackbone, self).__init__()
#         self.video_backbone = video_backbone
#         self.sketch_backbone = sketch_backbone

#     def forward(self, sketch_batch, video_batch):
#         '''
#         sketch_batch: [N, 1, C, H, W]
#         video_batch: [N, T, C, H, W]
#         '''
#         sketch_batch = sketch_batch.squeeze(1)  # (N, C, H, W)
#         src_sketch = self.sketch_backbone(sketch_batch).squeeze()  # (N, D)
#         src_sketch = src_sketch.unsqueeze(1)  # (N, 1, D)

#         video_batch = video_batch.transpose(1, 2)  # (N, C0, T0, H0, W0)
#         src_video = self.video_backbone(video_batch)  # (N, C, T, H, W)
#         src_video = src_video.flatten(2,-1)  # (N, C, T*H*W)
#         src_video = src_video.transpose(1, 2)  # (N, T*H*W, C)

#         return src_sketch, src_video


def build_backbone(args):
    if 'vit' in args.backbone:
        video_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        sketch_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        video_backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
        sketch_backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda()

        args.input_vid_dim = 768
        args.input_skch_dim = 768
    
        backbone = ViTBackbone(
            video_feature_extractor,
            sketch_feature_extractor,
            video_backbone,
            sketch_backbone,
        )
    elif 'resnet' in args.backbone:
        # video_backbone = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-2]) # before avgpool-fc
        sketch_backbone = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]) # before avgpool-fc

        video_backbone = nn.Sequential(*list(resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).children())[:-2]) # before avgpool-fc
        # sketch_backbone = nn.Sequential(*list(resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).children())[:-1]) # before avgpool-fc

        args.input_vid_dim = 512
        args.input_skch_dim = 512

        # video_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-2]) # before avgpool-fc
        # sketch_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-1]) # before avgpool-fc

        # args.input_vid_dim = 2048
        # args.input_skch_dim = 2048

        backbone = ResNetBackbone(
            video_backbone,
            sketch_backbone
        )
    # elif 's3d' in args.backbone:
    #     assert args.tight_frame_sampling == True
        
    #     video_backbone = nn.Sequential(*list(s3d(weights=S3D_Weights.KINETICS400_V1).children())[:-2]) # before avgpool-fc
    #     args.input_vid_dim = 1024

    #     sketch_backbone = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]) # before avgpool-fc
    #     args.input_skch_dim = 512

    #     backbone = S3DBackbone(
    #         video_backbone,
    #         sketch_backbone
    #     )
    else:
        raise NotImplementedError
    
    return backbone