from collections import Sequence
from typing import Tuple, Iterable, Union

import cv2
import numpy as np
from torch.utils.data.dataset import IterableDataset

import intelligent_placer_lib.dataset.functional as F


class SegmentationDataset(IterableDataset):

    def __init__(self, object_images: Sequence, object_masks: Sequence,
                 image_shape: Tuple, paper_shape: Tuple, border_spacing: int,
                 top_bottom_sep_y: int, num_objects: int, paper_color: Union,
                 polygon_spacing: int, polygon_size: int,
                 background_samples: np.ndarray, transform=None):
        self.object_images = object_images
        self.object_masks = object_masks
        self.image_shape = image_shape
        self.paper_shape = paper_shape
        self.border_spacing = border_spacing
        self.top_bottom_sep_y = top_bottom_sep_y
        self.num_objects = num_objects
        self.paper_color = paper_color
        self.polygon_spacing = polygon_spacing
        self.polygon_size = polygon_size
        self.background_samples = background_samples

        self.transform = transform

        assert len(self.object_images) == len(self.object_masks)

    def sample(self):
        image = F.make_background(self.background_samples, self.image_shape)
        F.make_random_stains(image)

        obj_indices = np.random.choice(range(len(self.object_images)), self.num_objects, replace=False)

        object_images = []
        object_masks = []

        for i in obj_indices:
            angle = np.random.randint(0, 360)
            mat = cv2.getRotationMatrix2D((self.object_images[i].shape[1] // 2, self.object_images[i].shape[0] // 2),
                                          angle, 1)

            obj = cv2.warpAffine(self.object_images[i], mat, self.object_images[i].shape[:2])
            mask = cv2.warpAffine(self.object_masks[i], mat, self.object_masks[i].shape[:2], cv2.INTER_NEAREST)

            nz_y, nz_x = np.nonzero(mask)

            x1, y1 = nz_x.min(), nz_y.min() + 1
            x2, y2 = nz_x.max(), nz_y.max() + 1

            object_images.append(obj[y1:y2, x1:x2])
            object_masks.append(mask[y1:y2, x1:x2])

        objects_bbox = [self.border_spacing, self.top_bottom_sep_y + self.border_spacing,
                        image.shape[1] - self.border_spacing, image.shape[0] - self.border_spacing]

        objects = F.sample_objects(object_images, object_masks, objects_bbox, self.border_spacing)

        for obj in objects:
            F.place_object(image, obj)

        paper_bbox = [self.border_spacing, self.border_spacing,
                      image.shape[1] - self.border_spacing, self.top_bottom_sep_y - self.border_spacing]

        paper = F.sample_paper(self.paper_shape, paper_bbox, self.paper_color)
        F.place_paper(image, paper)

        polygon_bbox = [paper.bbox[0] + self.polygon_spacing,
                        paper.bbox[1] + self.polygon_spacing,
                        paper.bbox[2] - self.polygon_spacing,
                        paper.bbox[3] - self.polygon_spacing]

        polygon = F.sample_polygon(self.polygon_size, polygon_bbox)
        F.place_polygon(image, polygon)

        image = F.add_light_gradient(image)

        image = np.float32(image) / 255

        object_boxes = [F.make_object_bbox(obj) for obj in objects]
        object_labels = [1] * len(objects)
        object_masks = [F.make_object_mask(self.image_shape, obj) for obj in objects]

        polygon_box = F.make_polygon_bbox(polygon)
        polygon_label = 2
        polygon_mask = F.make_polygon_mask(self.image_shape, polygon)

        bboxes = object_boxes + [polygon_box]
        labels = object_labels + [polygon_label]
        masks = object_masks + [polygon_mask]

        return {'image': image, 'bboxes': bboxes, 'labels': labels, 'masks': masks}

    def __iter__(self):
        return self

    def __next__(self):
        sample = self.sample()

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample
