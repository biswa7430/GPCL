import os

import torch
import torch.utils.data
import torchvision
from . import transforms as T
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


def validate_and_filter_boxes(targets, min_box_size=1.0):
    """
    Validate and filter out invalid bounding boxes from targets.
    
    This function removes boxes that have:
    - Zero or negative width/height
    - Width or height less than min_box_size
    - NaN or Inf coordinates
    
    Args:
        targets: List of target dictionaries containing 'boxes' and 'labels'
        min_box_size: Minimum valid box size in pixels (default: 1.0)
    
    Returns:
        List of filtered target dictionaries
    """
    filtered_targets = []
    
    for target in targets:
        if 'boxes' not in target or len(target['boxes']) == 0:
            filtered_targets.append(target)
            continue
        
        boxes = target['boxes']
        
        # Check for NaN or Inf
        valid_coords = ~(torch.isnan(boxes).any(dim=1) | torch.isinf(boxes).any(dim=1))
        
        # Check for positive width and height with minimum size
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        valid_size = (box_width >= min_box_size) & (box_height >= min_box_size)
        
        # Combine all validations
        keep = valid_coords & valid_size
        
        if keep.sum() == 0:
            # Skip samples with no valid boxes
            continue
        
        # Filter all target fields
        new_target = {}
        for key, value in target.items():
            if key == 'image_id':
                new_target[key] = value
            elif isinstance(value, torch.Tensor) and len(value) == len(boxes):
                new_target[key] = value[keep]
            else:
                new_target[key] = value
        
        filtered_targets.append(new_target)
    
    return filtered_targets


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


# class ConvertCocoPolysToMask:
#     def __call__(self, image, target):
#         w, h = image.size

#         image_id = target["image_id"]

#         anno = target["annotations"]

#         anno = [obj for obj in anno if obj["iscrowd"] == 0]

#         boxes = [obj["bbox"] for obj in anno]
#         # guard against no boxes via resizing
#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         boxes[:, 2:] += boxes[:, :2]
#         boxes[:, 0::2].clamp_(min=0, max=w)
#         boxes[:, 1::2].clamp_(min=0, max=h)

#         classes = [obj["category_id"] for obj in anno]
#         classes = torch.tensor(classes, dtype=torch.int64)

#         segmentations = [obj["segmentation"] for obj in anno]
#         masks = convert_coco_poly_to_mask(segmentations, h, w)

#         keypoints = None
#         if anno and "keypoints" in anno[0]:
#             keypoints = [obj["keypoints"] for obj in anno]
#             keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
#             num_keypoints = keypoints.shape[0]
#             if num_keypoints:
#                 keypoints = keypoints.view(num_keypoints, -1, 3)

#         keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
#         boxes = boxes[keep]
#         classes = classes[keep]
#         masks = masks[keep]
#         if keypoints is not None:
#             keypoints = keypoints[keep]

#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = classes
#         target["masks"] = masks
#         target["image_id"] = image_id
#         if keypoints is not None:
#             target["keypoints"] = keypoints

#         # for conversion to coco api
#         area = torch.tensor([obj["area"] for obj in anno])
#         iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
#         target["area"] = area
#         target["iscrowd"] = iscrowd

#         return image, target

# ...existing code...

class ConvertCocoPolysToMask:
    def __init__(self, min_box_size=1.0):
        """
        Args:
            min_box_size: Minimum width and height for valid boxes (default: 1.0 pixels)
        """
        self.min_box_size = min_box_size
    
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Only process segmentations if they exist
        if anno and "segmentation" in anno[0]:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        else:
            masks = None

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # Filter out invalid boxes with stricter validation
        # Keep boxes with positive height AND width AND meet minimum size requirement
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        keep = (box_height >= self.min_box_size) & (box_width >= self.min_box_size)
        
        boxes = boxes[keep]
        classes = classes[keep]
        if masks is not None:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if masks is not None:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        return image, target

# ...existing code...

def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": [], "info": {}}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    # FIXME: This is... awful?
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

        # Fix missing 'info' and 'licenses' fields in COCO dataset
        if 'info' not in self.coco.dataset:
            self.coco.dataset['info'] = {
                "description": "VisDrone Dataset",
                "url": "",
                "version": "1.0",
                "year": 2024,
                "contributor": "",
                "date_created": "2024/01/01"
            }
        if 'licenses' not in self.coco.dataset:
            self.coco.dataset['licenses'] = []

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False):
    """
    Load COCO-format dataset with support for both naming conventions:
    - Standard COCO: train2017/, instances_train2017.json
    - Simple format: train/, train.json
    """
    # Try simple format first (train/, train.json)
    simple_img_folder = os.path.join(root, image_set)
    simple_ann_file = os.path.join(root, "annotations", f"{image_set}.json")
    
    # Try standard COCO format (train2017/, instances_train2017.json)
    anno_file_template = "{}_{}2017.json"
    coco_img_folder = os.path.join(root, f"{image_set}2017")
    coco_ann_file = os.path.join(root, "annotations", anno_file_template.format(mode, image_set))
    
    # Determine which format to use
    if os.path.exists(simple_ann_file):
        # Use simple format
        img_folder = simple_img_folder
        ann_file = simple_ann_file
    elif os.path.exists(coco_ann_file):
        # Use COCO format
        img_folder = coco_img_folder
        ann_file = coco_ann_file
    else:
        # Fallback to simple format (will raise error if not found)
        img_folder = simple_img_folder
        ann_file = simple_ann_file

    if use_v2:
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
        target_keys = ["boxes", "labels", "image_id"]
        if with_masks:
            target_keys += ["masks"]
        dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=target_keys)
    else:
        # TODO: handle with_masks for V1?
        t = [ConvertCocoPolysToMask()]
        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)

        dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def collate_fn_with_validation(batch, min_box_size=1.0):
    """
    Safe collate function that validates and filters invalid boxes.
    
    This prevents training crashes due to degenerate bounding boxes
    that can occur from augmentation or annotation errors.
    
    Args:
        batch: List of (image, target) tuples
        min_box_size: Minimum box size in pixels (default: 1.0)
    
    Returns:
        Tuple of (images, targets) with validated boxes
    """
    batch = list(zip(*batch))
    images = batch[0]
    targets = batch[1]
    
    # Validate each target individually and track valid indices
    filtered_images = []
    filtered_targets = []
    
    for i, (image, target) in enumerate(zip(images, targets)):
        if 'boxes' not in target or len(target['boxes']) == 0:
            # Keep samples without boxes (will be handled later)
            filtered_images.append(image)
            filtered_targets.append(target)
            continue
        
        boxes = target['boxes']
        
        # Check for NaN or Inf
        valid_coords = ~(torch.isnan(boxes).any(dim=1) | torch.isinf(boxes).any(dim=1))
        
        # Check for positive width and height with minimum size
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        valid_size = (box_width >= min_box_size) & (box_height >= min_box_size)
        
        # Combine all validations
        keep = valid_coords & valid_size
        
        if keep.sum() == 0:
            # Skip samples with no valid boxes
            continue
        
        # Filter all target fields
        new_target = {}
        for key, value in target.items():
            if key == 'image_id':
                new_target[key] = value
            elif isinstance(value, torch.Tensor) and len(value) == len(boxes):
                new_target[key] = value[keep]
            else:
                new_target[key] = value
        
        filtered_images.append(image)
        filtered_targets.append(new_target)
    
    # If all samples were filtered out, return empty batch
    if len(filtered_images) == 0:
        return [], []
    
    return tuple(filtered_images), tuple(filtered_targets)
