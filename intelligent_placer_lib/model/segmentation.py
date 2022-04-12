from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class SegmentationModelPL(pl.LightningModule):

    def __init__(self, input_shape: Tuple):
        super(SegmentationModelPL, self).__init__()

        self.model = MaskRCNN(resnet_fpn_backbone('resnet34', pretrained=True, trainable_layers=5),
                              num_classes=3,
                              min_size=min(input_shape),
                              max_size=max(input_shape),
                              image_mean=[0, 0, 0],
                              image_std=[1, 1, 1])

    def training_step(self, batch, batch_idx):
        xs = [torch.as_tensor(sample['image'], device=self.device) for sample in batch]
        ys = [dict(boxes=torch.tensor(np.array(sample['bboxes']), device=self.device, dtype=torch.float32),
                   labels=torch.tensor(np.array(sample['labels']), device=self.device, dtype=torch.int64),
                   masks=torch.tensor(np.array(sample['masks']), device=self.device)) for sample in batch]

        loss_dict = self.model.train()(xs, ys)
        loss = sum(v for k, v in loss_dict.items())

        self.log('loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
