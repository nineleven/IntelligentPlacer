from typing import Tuple, List

import cv2
import numpy as np
from shapely.geometry import Polygon
from torch.utils.data import IterableDataset

import intelligent_placer_lib.dataset.functional as F


class PlacerRandomDataset(IterableDataset):

    def __init__(self, small_poly_num_vertices: int, large_poly_num_vertices: int,
                 small_poly_max_size, large_poly_max_size: int, image_size: int,
                 poly_center_range: Tuple):
        self.small_poly_num_vertices = small_poly_num_vertices
        self.large_poly_num_vertices = large_poly_num_vertices
        self.small_poly_max_size = small_poly_max_size
        self.large_poly_max_size = large_poly_max_size
        self.image_size = image_size
        self.poly_center_range = poly_center_range

    def random_polygon(self, num_vertices: int, max_size: int):
        image = np.zeros([self.image_size, self.image_size], dtype=np.float32)

        coords = max_size * np.array(F.to_convex_contour(num_vertices))

        center = np.random.randint(*self.poly_center_range, size=2).reshape(1, 2)

        coords = np.int32(coords - coords.mean(axis=0, keepdims=True) + center)

        image = cv2.fillPoly(image, [coords], 1)

        return image, (center.squeeze() - self.image_size // 2) / (self.image_size // 2)

    def __next__(self):
        small_polygons = []
        means = []

        for _ in range(3):
            poly, mean = self.random_polygon(self.small_poly_num_vertices, self.small_poly_max_size)
            small_polygons.append(poly)
            means.append(mean)

        x = np.stack(small_polygons)
        y, _ = self.random_polygon(self.small_poly_num_vertices, self.large_poly_max_size)
        means = np.stack(means)

        return x, y[np.newaxis, ...], np.float32(means)

    def __iter__(self):
        return self


class PlacerDoableDataset(IterableDataset):

    def __init__(self, large_polygon_num_vertices: int, large_polygon_max_size: int,
                 poly_center_range: Tuple, image_size: int):
        self.large_polygon_num_vertices = large_polygon_num_vertices
        self.large_polygon_max_size = large_polygon_max_size
        self.image_size = image_size
        self.poly_center_range = poly_center_range

    def split_polygon(self, hull):
        sp1 = len(hull) // 3
        sp2 = 2 * len(hull) // 3

        center = (hull[0] + hull[sp1] + hull[sp2]) // 3

        a = np.int32(0.1 * (hull.min(axis=0) - center))
        b = np.int32(0.1 * (hull.max(axis=0) - center))
        b = np.maximum(b, a + 1)

        new_point = center + np.random.randint(a, b)

        hull1 = np.concatenate([hull[-1][np.newaxis, ...], hull[:sp1], new_point[np.newaxis, ...]])
        hull2 = np.concatenate([hull[sp1 - 1:sp2], new_point[np.newaxis, ...]])
        hull3 = np.concatenate([hull[sp2 - 1:], new_point[np.newaxis, ...]])

        return hull1, hull2, hull3

    def random_transform(self, hull):
        hull_mean = np.mean(hull, axis=0, keepdims=True)

        hull -= hull_mean

        hull = np.concatenate([hull, np.ones([len(hull), 1])], axis=1)

        s1, s2 = 0.7 + 0.3 * np.random.random(2)
        r1, r2 = (np.random.random(2) - 0.5) / 5
        mat = np.array([[s1, r1, 0], [r2, s2, 0], [0, 0, 1]])

        hull = hull.dot(mat.T)

        theta = -np.pi / 6 + 2 * np.pi / 6 * np.random.random()

        mat = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])

        hull = hull.dot(mat.T)

        hull += hull_mean

        return np.int32(hull)

    def make_image(self, coords):
        image = np.zeros([self.image_size, self.image_size], dtype=np.float32)

        center = np.random.randint(*self.poly_center_range, size=2).reshape(1, 2)

        coords = np.int32(coords - coords.mean(axis=0, keepdims=True) + center)

        image = cv2.fillPoly(image, [coords], 1)

        return image, (center.squeeze() - self.image_size // 2) / (self.image_size // 2)

    def __next__(self):
        hull_y = self.large_polygon_max_size * np.array(F.to_convex_contour(self.large_polygon_num_vertices))
        hull_y = (hull_y - np.mean(hull_y, axis=0, keepdims=True) + self.image_size // 2).astype(np.int32)

        hulls_x = self.split_polygon(hull_y)

        hulls_x = [self.random_transform(hulls_x[i]) for i in range(len(hulls_x))]

        images_x = []
        means_x = []

        for i in range(len(hulls_x)):
            img, mean = self.make_image(hulls_x[i])
            images_x.append(img)
            means_x.append(mean)

        images_x = np.stack(images_x)
        means_x = np.stack(means_x)

        image_y, _ = self.make_image(hull_y)

        return images_x, image_y[np.newaxis, ...], np.float32(means_x)

    def __iter__(self):
        return self


class PlacerRealDataset(IterableDataset):

    def __init__(self, polygons: List, image_size: int):
        self.polygons = polygons
        self.image_size = image_size

    def get_intersection_matrix(self, polygons):
        mat = np.zeros((len(polygons), len(polygons)))

        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                mat[i, j] = mat[j, i] = Polygon(polygons[i]).intersects(Polygon(polygons[j]))

        return mat

    def fix(self, polygons):
        step = 5

        mat = self.get_intersection_matrix(polygons)

        while np.count_nonzero(mat):
            poly_centers = np.array(list(map(lambda x: np.mean(x, axis=0), polygons)))

            for i in range(len(polygons)):
                intersects_with = np.nonzero(mat[i])[0]

                if len(intersects_with) > 0:
                    direction = poly_centers[i] - np.mean(poly_centers[intersects_with], axis=0)
                    direction = direction / np.sum(direction ** 2) ** 0.5

                    polygons[i] = cv2.convexHull(polygons[i] + np.int32(direction[np.newaxis, :] * step))[:, 0]

            mat = self.get_intersection_matrix(polygons)
        return polygons

    def random_transform(self, hull):
        hull_mean = np.mean(hull, axis=0, keepdims=True)

        hull = hull - hull_mean

        hull = np.concatenate([hull, np.ones([len(hull), 1])], axis=1)

        s1, s2 = 0.7 + 0.3 * np.random.random(2)
        r1, r2 = (np.random.random(2) - 0.5) / 5
        mat = np.array([[s1, r1, 0], [r2, s2, 0], [0, 0, 1]])

        hull = hull.dot(mat.T)

        theta = -np.pi / 6 + 2 * np.pi / 6 * np.random.random()

        mat = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])

        hull = hull.dot(mat.T)

        hull = hull + hull_mean

        return np.int32(hull)

    def __next__(self):
        poly_subset_indices = np.random.choice(range(len(self.polygons)), 3, replace=False)
        polys = []

        for i in poly_subset_indices:
            poly = self.polygons[i].copy()
            poly_mean = self.image_size // 2 + np.random.randint(-40, 41, size=2).reshape(1, 2)
            poly = poly + poly_mean - poly.mean(axis=0, keepdims=True)
            polys.append(np.int32(poly))

        polys = self.fix(polys)

        hull = cv2.convexHull(np.concatenate(polys, axis=0))[:, 0]
        hull = np.int32(hull - np.mean(hull, axis=0, keepdims=True) + self.image_size // 2)

        polys = [
            p + self.image_size // 2 + np.random.randint(-40, 41, size=2).reshape(1, 2) - p.mean(axis=0, keepdims=True)
            for p in polys]
        polys = [np.int32(self.random_transform(p)) for p in polys]
        poly_means = [(np.mean(p, axis=0) - self.image_size // 2) / (self.image_size // 2) for p in polys]

        polys = np.stack(
            [cv2.fillPoly(np.zeros([self.image_size, self.image_size], dtype=np.float32), [p], 1) for p in polys])
        hull = cv2.fillPoly(np.zeros([self.image_size, self.image_size], dtype=np.float32), [hull], 1)[np.newaxis, ...]

        return polys, hull, np.float32(np.stack(poly_means))

    def __iter__(self):
        return self
