from dataclasses import dataclass
from typing import Tuple, Iterable, Union, Sequence, List

import cv2
import numpy as np
from matplotlib import pyplot as plt

from intelligent_placer_lib.dataset.polygon_generation import to_convex_contour


@dataclass
class Object:
    x: int
    y: int

    image: np.ndarray
    mask: np.ndarray


@dataclass
class Paper:
    bbox: Sequence[int]
    color: Union[int, Iterable]


@dataclass
class Polygon:
    coords: np.ndarray


def wrap(texture, size):
    img = np.zeros((*size, 3), dtype=np.uint8)

    for i in range(0, size[0], texture.shape[0]):
        for j in range(0, size[1], texture.shape[1]):
            tile_size_x = min(texture.shape[0], size[0] - i)
            tile_size_y = min(texture.shape[1], size[1] - j)

            w = texture

            if (i // texture.shape[0]) % 2 == 1:
                w = w[::-1]

            if (j // texture.shape[1]) % 2 == 1:
                w = w[:, ::-1]

            w = w[:tile_size_x, :tile_size_y]

            img[i:i + tile_size_x, j:j + tile_size_y] = w

    return img


def random_crop(img, size):
    x1 = np.random.randint(0, img.shape[0] - size[0])
    y1 = np.random.randint(0, img.shape[1] - size[1])

    return img[x1:x1 + size[0], y1:y1 + size[1]]


def make_background(samples: List, size: Tuple):
    sample = np.random.choice(samples)
    crop = random_crop(sample, (170, 170))
    return wrap(crop, size)


def sample_hspaces(image_width, obj_widths, min_space_between):
    random_spaces_sum = image_width

    for i in range(len(obj_widths)):
        if obj_widths[i] + min_space_between > random_spaces_sum:
            break
        random_spaces_sum -= obj_widths[i] + min_space_between

    space_coefs = np.random.random(i)
    hspaces = min_space_between + np.int32(space_coefs / sum(space_coefs) * random_spaces_sum)

    return hspaces


def sample_objects(object_images, object_masks, bbox, min_space_between):
    obj_heights = [obj.shape[0] for obj in object_images]
    obj_widths = [obj.shape[1] for obj in object_images]

    hspaces = sample_hspaces(bbox[2] - bbox[0], obj_widths, min_space_between)

    offsets_x = bbox[0] + np.cumsum(hspaces + np.array([0, *obj_widths[:len(hspaces) - 1]]))

    min_offset_y = bbox[1]
    max_offsets_y = bbox[3] - np.array(obj_heights)
    offsets_y = np.random.randint(min_offset_y, max_offsets_y)

    objects = []

    for obj_img, obj_mask, offset_x, offset_y in zip(object_images, object_masks, offsets_x, offsets_y):
        objects.append(Object(offset_x, offset_y, obj_img, obj_mask))

    return objects


def sample_polygon(polygon_size, bbox):
    polygon_x1 = np.random.randint(bbox[0], bbox[2] - polygon_size)
    polygon_y1 = np.random.randint(bbox[1], bbox[3] - polygon_size)

    unitary_polygon = np.array(to_convex_contour(10))
    coords = np.int32(np.array([[polygon_x1, polygon_y1]]) + polygon_size * unitary_polygon)

    return Polygon(coords)


def sample_paper(paper_shape, bbox, color):
    x1 = np.random.randint(bbox[0], bbox[2] - paper_shape[1])
    y1 = np.random.randint(bbox[1], bbox[3] - paper_shape[0])

    return Paper([x1, y1, x1 + paper_shape[1], y1 + paper_shape[0]], color)


def place_object(image: np.ndarray, obj: Object):
    x1, y1 = obj.x, obj.y
    x2, y2 = x1 + obj.image.shape[1], y1 + obj.image.shape[0]
    image[y1:y2, x1:x2] = np.where(obj.mask[..., np.newaxis], obj.image, image[y1:y2, x1:x2])


def place_paper(image, paper: Paper):
    image[paper.bbox[1]: paper.bbox[3], paper.bbox[0]: paper.bbox[2]] = paper.color


def place_polygon(image, polygon: Polygon):
    thickness = np.random.randint(1, 5)
    cv2.drawContours(image, [polygon.coords], contourIdx=0, color=0, thickness=thickness)


def make_random_stains(background_image: np.ndarray):
    num_stains = np.random.randint(0, 7)

    for _ in range(num_stains):
        stain_size = 2 * np.random.randint(4, 7) + 1
        x, y = np.random.randint(stain_size // 2, np.array(background_image.shape[:2]) - stain_size // 2 - 1, size=2)
        color = np.random.randint(10, 50) * np.ones((1, 1, 3))

        mx, my = np.meshgrid(range(stain_size), range(stain_size))
        stain_img = np.clip(
            np.exp(0.005 * ((mx - stain_size // 2) ** 2 + (my - stain_size // 2) ** 2)).reshape(stain_size, stain_size,
                                                                                              1) * color, 0,
            255).astype(np.uint8)
        background_image[x - stain_size // 2:x + stain_size // 2 + 1,
        y - stain_size // 2:y + stain_size // 2 + 1] = stain_img


def make_object_bbox(obj: Object) -> np.ndarray:
    return np.array([obj.x, obj.y, obj.x + obj.image.shape[1], obj.y + obj.image.shape[0]])


def make_polygon_bbox(polygon: Polygon) -> np.ndarray:
    x1, y1 = np.min(polygon.coords, axis=0)
    x2, y2 = np.max(polygon.coords, axis=0)

    return np.array([x1, y1, x2, y2])


def make_object_mask(image_shape: Tuple[int, ...], obj: Object) -> np.ndarray:
    mask = np.zeros(image_shape)

    mask[obj.y:obj.y + obj.image.shape[0], obj.x:obj.x + obj.image.shape[1]] = obj.mask

    return mask


def make_polygon_mask(image_shape: Tuple[int, ...], polygon: Polygon) -> np.ndarray:
    mask = np.zeros(image_shape)

    mask = cv2.fillPoly(mask, [polygon.coords], 1)

    return mask


def add_light_gradient(image: np.ndarray) -> np.ndarray:
    direction = 2 * np.pi * np.random.random()
    intensity = np.int32(np.array([50, 25, 5]) * np.random.random(size=3)).reshape(1, 1, -1)

    x, y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))

    proj = (x * np.cos(direction) + y * np.sin(direction)) ** 3

    lighting = (intensity * (proj[..., np.newaxis] - proj.min()) / (proj.max() - proj.min())).astype(np.uint8)

    new_image = image.astype(np.int32) + lighting.astype(np.int32)

    return (new_image / new_image.max() * 255).astype(np.uint8)
