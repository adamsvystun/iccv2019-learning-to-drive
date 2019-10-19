from collections import defaultdict, OrderedDict
from typing import Dict, Set

from torchvision import models
import torch.nn as nn
import torch


class SlowFusionModel(nn.Module):

    def __init__(self, cameras: Set[str], num_frames_per_camera: int):
        super().__init__()
        self.cameras = cameras
        self.num_frames_per_camera = num_frames_per_camera

        # Build towers to extract features from each frame from each camera
        self.towers: Dict[str, Dict[int, nn.Module]] = defaultdict(dict)
        for camera in self.cameras:
            for frame in range(self.num_frames_per_camera):
                resnet = models.resnet18(pretrained=True)
                self.towers[camera][frame] = nn.Sequential(*list(resnet.children())[:6])

        # Fuse features of each two subsequent frames for each camera
        self.early_fusions: Dict[str, Dict[str, nn.Module]] = defaultdict(dict)
        for camera in self.cameras:
            for frame in range(self.num_frames_per_camera - 1):
                frame_pair = str(frame) + str(frame + 1)
                self.early_fusions[camera][frame_pair] = self._make_early_fusion()

        # Fuse volumes from each two fused subsequent frames for each camera
        self.middle_fusions = dict()
        for camera in self.cameras:
            self.middle_fusions[camera] = self._make_middle_fusion()

        # Fuse volume from each camera
        self.late_fusion = self._make_late_fusion()

        # Build two regression heads for each target
        self.speed_head = self._make_regression_head()
        self.angle_head = self._make_regression_head()

    def forward(self, x: Dict[str, Dict[int, torch.Tensor]]):
        # Extract features from each frame from each camera
        towers_out = {camera: {frame: self.towers[camera].get(frame)(x[camera][frame]) for frame in frames}
                      for camera, frames in self.towers.items()}

        # Fuse features of each two subsequent frames for each camera
        early_fusions_out: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        for camera in self.cameras:
            for frame in range(self.num_frames_per_camera - 1):
                frames = towers_out[camera][frame], towers_out[camera][frame + 1]
                frames = torch.cat((frames[0], frames[1]), 1)
                frame_pair = str(frame) + str(frame + 1)
                early_fusions_out[camera][frame_pair] = self.early_fusions[camera].get(frame_pair)(frames)

        # Fuse volumes from each two fused subsequent frames for each camera
        middle_fusions_out = dict()
        for camera in self.cameras:
            frames_pairs = tuple(list(early_fusions_out[camera].values()))
            frames_pairs = torch.cat(frames_pairs, 1)
            middle_fusions_out[camera] = self.middle_fusions[camera](frames_pairs)

        # Fuse volume from each camera
        cameras = tuple(middle_fusions_out.values())
        cameras = torch.cat(cameras, 1)
        late_fusion_out = self.late_fusion(cameras)

        # Perform prediction on both regression heads
        late_fusion_out = late_fusion_out.view(late_fusion_out.size(0), -1)
        speed_head_out = self.speed_head(late_fusion_out)
        angle_head_out = self.angle_head(late_fusion_out)

        return {'canSpeed': torch.squeeze(speed_head_out),
                'canSteering': torch.squeeze(angle_head_out)}

    def cuda(self, device: str = None):
        super().cuda()
        for camera in self.cameras:
            for frame in range(self.num_frames_per_camera):
                self.towers[camera][frame].cuda()
        for camera in self.cameras:
            for frame in range(self.num_frames_per_camera - 1):
                frame_pair = str(frame) + str(frame + 1)
                self.early_fusions[camera][frame_pair].cuda()
        for camera in self.cameras:
            self.middle_fusions[camera].cuda()
        return self

    def _make_early_fusion(self) -> nn.Module:
        return nn.Sequential(OrderedDict({
            'block1': self._make_block(256, 256),
            'block2': self._make_block(256, 128),
            'pool': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

    def _make_middle_fusion(self) -> nn.Module:
        num_filters = 128 * (self.num_frames_per_camera - 1)
        return nn.Sequential(OrderedDict({
            'block1': self._make_block(num_filters, 512),
            'block2': self._make_block(512, 256),
            'block3': self._make_block(256, 128),
            'pool': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

    def _make_late_fusion(self) -> nn.Module:
        num_filters = 128 * len(self.cameras)
        return nn.Sequential(OrderedDict({
            'block1': self._make_block(num_filters, num_filters),
            'block2': self._make_block(num_filters, 256),
            'pool': nn.MaxPool2d(kernel_size=2, stride=2),
            'block4': self._make_block(256, 512),
        }))

    def _make_regression_head(self) -> nn.Module:
        return nn.Sequential(OrderedDict({
            'lin1': nn.Linear(5120, 64),
            'relu1': nn.ReLU(),
            'lin2': nn.Linear(64, 32),
            'relu2': nn.ReLU(),
            'lin3': nn.Linear(32, 1),
        }))

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            'bn': nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            'relu': nn.ReLU(),
        }))
