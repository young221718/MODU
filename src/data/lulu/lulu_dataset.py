"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data
import torchvision
from torchvision.tv_tensors import BoundingBoxes, Mask, BoundingBoxFormat
from pycocotools import mask as coco_mask
import pandas as pd
from PIL import Image
import os
from typing import Any, Callable, List, Optional, Tuple, Union

from src.core import register

__all__ = ["LuluDetection"]


@register
class LuluDetection(torchvision.datasets.CocoDetection):
    __inject__ = ["transforms"]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
    ):
        super(LuluDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMaskWithFacebox(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks

    def _load_face_box(self, id: int) -> List:
        face_box = self.coco.loadImgs(id)[0]["face_box"]
        return face_box

    def __getitem__(self, idx):
        img, target = super(LuluDetection, self).__getitem__(idx)
        face_box = self._load_face_box(idx)

        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target, face_box)

        # ['boxes', 'masks', 'labels']:
        if "boxes" in target:
            target["boxes"] = BoundingBoxes(
                target["boxes"],
                format=BoundingBoxFormat.XYXY,
                canvas_size=img.size[::-1],
            )  # h w

        if "masks" in target:
            target["masks"] = Mask(target["masks"])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"

        return s


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMaskWithFacebox(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target, facebox):
        w, h = facebox[:2]
        facebox[2], facebox[3] = facebox[2] + w, facebox[3] + h
        print(facebox)
        cropped_image = image.crop(facebox)

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:2] -= [w, h]  # face box 부분 처리리
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:  # TODO: facebox 고려해야함함
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None  # TODO: facebox 고려해야함함
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])

        return cropped_image, target
