import os
from dataclasses import dataclass
from typing import Union, List, Dict

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import rotate
from torch import nn
from torchvision.ops import nms

from intelligent_placer_lib.model.placer import PlacerResnet
from intelligent_placer_lib.model.segmentation import SegmentationModelPL


@dataclass
class SegmentationResults:
    polygon_mask: np.ndarray
    object_masks: List


@dataclass
class PlacerResults:
    object_masks_transformed: np.ndarray


def load_placer_model(path: str) -> nn.Module:
    model = PlacerResnet()
    model.load_state_dict(torch.load(path))
    return model


def load_segmentation_model(path: str) -> nn.Module:
    net = SegmentationModelPL((640, 360))
    net.model.load_state_dict(torch.load(path))

    return net.model


def expand_mask(mask, target_size):
    expanded_mask = np.zeros((target_size,
                              target_size, 1),
                             dtype=np.float32)

    x1 = target_size // 2 - mask.shape[1] // 2
    y1 = target_size // 2 - mask.shape[0] // 2
    x2 = x1 + mask.shape[1]
    y2 = y1 + mask.shape[0]

    expanded_mask[y1:y2, x1:x2] = mask

    return expanded_mask


def cut_out_bbox(mask, bbox) -> np.ndarray:
    return mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def make_segmentation_results(res: Dict) -> SegmentationResults:
    res['masks'] = (res['masks'] > 0.5).type(torch.float32)

    idx = nms(res['boxes'], res['scores'], 0.1)
    idx = idx[res['scores'][idx] > 0.1]

    res['boxes'] = res['boxes'][idx]
    res['scores'] = res['scores'][idx]
    res['labels'] = res['labels'][idx]
    res['masks'] = res['masks'][idx]

    res = {k: v.detach().numpy() for k, v in res.items()}
    res['masks'] = np.array([np.moveaxis(mask, 0, -1) for mask in res['masks']])
    res['boxes'] = res['boxes'].astype(np.int32)

    assert 1 in res['labels'] and (res['labels'] == 2).sum() == 1

    object_masks = [cut_out_bbox(mask, box) for mask, box in zip(res['masks'][res['labels'] == 1],
                                                                 res['boxes'][res['labels'] == 1])]
    polygon_mask = cut_out_bbox(res['masks'][res['labels'] == 2][0],
                                res['boxes'][res['labels'] == 2][0])

    return SegmentationResults(polygon_mask, object_masks)


def compute_loss(polygon_mask, object_masks):
    def jaccard_loss(inputs, targets):
        inputs = inputs.flatten()
        targets = targets.flatten()

        numer = 2 * inputs * targets
        denom = inputs + targets

        return np.sum(numer) / (1e-6 + np.sum(denom))

    def my_loss(inputs, targets):
        inputs = inputs.flatten()
        targets = targets.flatten()

        numer = inputs * targets
        denom = inputs

        return 1 - np.sum(numer) / (1e-6 + np.sum(denom))

    object_masks = object_masks[..., np.any(object_masks != 0, axis=(0, 1))]

    pos_part = sum(my_loss(object_masks[..., k: k + 1], polygon_mask) for k in range(object_masks.shape[-1]))
    neg_part = sum(jaccard_loss(object_masks[..., i], object_masks[..., j])
                   for i in range(object_masks.shape[-1]) for j in range(i + 1, object_masks.shape[-1]) if
                   i != j)

    return pos_part + neg_part


def transform_masks(object_masks):
    transformed_masks = []

    for mask in object_masks:
        angle = np.random.randint(360)
        mask = rotate(mask, angle, order=0)
        sx, sy = np.random.randint(-20, 21, size=2)
        mask = np.roll(mask, (sx, sy), axis=(0, 1))
        transformed_masks.append(mask)

    return transformed_masks


class IntelligentPlacer:

    def __init__(self, segmentation_weights_path: str,
                 placer_weights_path: str, placer_image_size=256):
        self.segmentation_model = load_segmentation_model(segmentation_weights_path)
        self.placer_model = load_placer_model(placer_weights_path)
        self.placer_image_size = placer_image_size

    def run_segmentation(self, img: np.ndarray) -> SegmentationResults:
        img = np.float32(img / 255)
        img_tensor = torch.moveaxis(torch.as_tensor(img), -1, 0)

        res = self.segmentation_model.eval()([img_tensor])[0]

        return make_segmentation_results(res)

    def run_placer(self, object_masks: List, polygon_mask: np.ndarray) -> PlacerResults:
        img_x = torch.cat([torch.moveaxis(torch.as_tensor(mask), -1, 0) for mask in object_masks])
        img_y = torch.moveaxis(torch.as_tensor(polygon_mask), -1, 0)

        img_x = torch.cat([img_x, torch.zeros(3 - img_x.shape[0], *img_x.shape[1:])], dim=0)

        means = torch.zeros(3, 2)

        img_x_tr, _ = self.placer_model.eval()(img_x[None, ...], img_y[None, ...], means[None, ...])
        img_x_tr = img_x_tr[0].detach()

        return PlacerResults(np.moveaxis(img_x_tr.numpy(), 0, -1))

    def run(self, img: np.ndarray):
        seg_res = self.run_segmentation(img)

        object_masks = [expand_mask(obj_mask, self.placer_image_size) for obj_mask in seg_res.object_masks]
        polygon_mask = expand_mask(seg_res.polygon_mask, self.placer_image_size)

        losses = []
        min_loss = float('inf')
        best_transformed = None

        for _ in range(30):
            transformed_masks = transform_masks(object_masks)
            placer_res = self.run_placer(transformed_masks, polygon_mask)
            loss = compute_loss(polygon_mask, placer_res.object_masks_transformed)
            losses.append(loss)
            if loss < min_loss:
                min_loss = loss
                best_transformed = placer_res.object_masks_transformed

        result = 0.2 * np.repeat(polygon_mask, 3, axis=2)
        result = np.maximum(result, best_transformed)

        return min_loss, result


def check_image(path: Union[str, os.PathLike[str]]):
    img = io.imread(path)
    img = cv2.resize(img, (360, 640), interpolation=cv2.INTER_LINEAR)

    plt.subplot(121)
    plt.imshow(img)

    placer = IntelligentPlacer('segmentation_weights.pth',
                               'placer_weights.pth')

    loss, res = placer.run(img)

    print('Loss:', loss)

    plt.subplot(122)
    plt.imshow(res)
    plt.show()

    return loss < 5e-2


if __name__ == '__main__':
    check_image('data/real_images/photo5375590401093778254.jpg')
