from copy import deepcopy
from typing import Union, Optional, List, Dict

import numpy as np
import tensorflow as tf
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format, xywh_to_xyxy_format
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from numpy._typing import NDArray
from yolo_od.config import CONFIG
from yolo_od.yolo_helpers.yolo_utils import DECODER, DEFAULT_BOXES


def get_predict_bbox_list(data: tf.Tensor) -> List[BoundingBox]:
    """
    Description: This function takes a TensorFlow tensor data as input and returns a list of bounding boxes representing predicted annotations.
    Input: data (tf.Tensor): A TensorFlow tensor representing the output data.
    Output: bb_object (List[BoundingBox]): A list of bounding box objects representing the predicted annotations.
    """
    from_logits = True if CONFIG['MODEL_FORMAT'] != "inference" else False
    decoded = False if CONFIG['MODEL_FORMAT'] != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(
        np.reshape(data, (1, *data.shape)), decoded=decoded, image_size=CONFIG['IMAGE_SIZE'])
    # add batch
    outputs = DECODER(loc_list_reshaped,
                      class_list_reshaped,
                      DEFAULT_BOXES,
                      from_logits=from_logits,
                      decoded=decoded,
                      )
    bb_object = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=CONFIG['BACKGROUND_LABEL'])
    return bb_object


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    bb_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = np.array(bb_array)
    # fig, ax = plt.subplots(figsize=(6, 9)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][1:5])
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=CONFIG['CATEGORIES'][int(bb_array[i][min(5, len(bb_array[i]) - 1)])])

            bb_list.append(curr_bb)
    return bb_list


def polygon_to_bbox(vertices):
    """
    Converts a polygon representation to a bounding box representation.

    Args:
        vertices (list): List of vertices defining the polygon. The vertices should be in the form [x1, y1, x2, y2, ...].

    Returns:
        list: Bounding box representation of the polygon in the form [x, y, width, height].

    Note:
        - The input list of vertices should contain x and y coordinates in alternating order.
        - The function calculates the minimum and maximum values of the x and y coordinates to determine the bounding box.
        - The bounding box representation is returned as [x, y, width, height], where (x, y) represents the center point of the
          bounding box, and width and height denote the size of the bounding box.
    """

    xs = [x for i, x in enumerate(vertices) if i % 2 == 0]
    ys = [x for i, x in enumerate(vertices) if i % 2 != 0]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    # Bounding box representation: (x, y, width, height)
    bbox = [(min_x + max_x) / 2., (min_y + max_y) / 2., max_x - min_x, max_y - min_y]

    return bbox


def calculate_iou_all_pairs(bboxes: np.ndarray, image_size: int) -> np.ndarray:
    """
    Calculates the Intersection over Union (IOU) for all pairs of bounding boxes.

    This function utilizes vectorization to efficiently compute the IOU for all possible pairs of bounding boxes.
    By leveraging NumPy's array operations, the calculations are performed in parallel, leading to improved performance.

    Args:
        bboxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h].
        image_size (int): Size of the image.

    Returns:
        np.ndarray: Array containing the IOU values for all pairs of bounding boxes.
    """

    # Reformat all bboxes to (x_min, y_min, x_max, y_max)
    bboxes = np.asarray([xywh_to_xyxy_format(bbox[:-1]) for bbox in bboxes]) * image_size
    num_bboxes = len(bboxes)
    # Calculate coordinates for all pairs
    x_min = np.maximum(bboxes[:, 0][:, np.newaxis], bboxes[:, 0])
    y_min = np.maximum(bboxes[:, 1][:, np.newaxis], bboxes[:, 1])
    x_max = np.minimum(bboxes[:, 2][:, np.newaxis], bboxes[:, 2])
    y_max = np.minimum(bboxes[:, 3][:, np.newaxis], bboxes[:, 3])

    # Calculate areas for all pairs
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Calculate intersection area for all pairs
    intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

    # Calculate union area for all pairs
    union_area = areas[:, np.newaxis] + areas - intersection_area

    # Calculate IOU for all pairs
    iou = intersection_area / union_area
    iou = iou[np.triu_indices(num_bboxes, k=1)]
    return iou


def count_obj_bbox_occlusions(img: np.ndarray, bboxes: np.ndarray, occlusion_threshold: float, calc_avg_flag: bool) -> \
        Union[float, int]:
    """
    Counts the occluded bounding boxes of a specific object category in an image.

    This function takes an image and an array of bounding boxes as input and counts the number of occluded
    bounding boxes of a specific object category. The occlusion is determined based on the Intersection over Union (IOU)
    between the bounding boxes.

    Args:
        img (np.ndarray): Image represented as a NumPy array.
        bboxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h, label].
        occlusion_threshold (float): Threshold value for determining occlusion based on IOU.
        calc_avg_flag (bool): Flag indicating whether to calculate the average occlusion count.

    Returns:
        Union[float, int]: Number of occluded bounding boxes of the specified object category.
                           If calc_avg_flag is True, it returns the average occlusion count as a float.
                           If calc_avg_flag is False, it returns the total occlusion count as an integer.

    """
    img_size = img.shape[0]
    label = CONFIG["CATEGORIES"].index('object')
    obj_bbox = bboxes[bboxes[..., -1] == label]
    if len(obj_bbox) == 0:
        return 0.0
    else:
        ious = calculate_iou_all_pairs(obj_bbox, img_size)
        occlusion_count = len(ious[ious > occlusion_threshold])
        if calc_avg_flag:
            return int(occlusion_count / len(obj_bbox))
        else:
            return occlusion_count


def remove_label_from_bbs(bbs_object_array, removal_label, add_to_label):
    new_bb_arr = []
    for bb in bbs_object_array:
        if bb.label != removal_label:
            new_bb = deepcopy(bb)
            new_bb.label = new_bb.label + "_" + add_to_label
            new_bb_arr.append(new_bb)
    return new_bb_arr


def calculate_overlap(box1, box2):
    # Extract coordinates of the bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate the overlap area
    overlap_area = w_intersection * h_intersection

    return overlap_area


def extract_bboxes_yolo(path: str):
    bboxes = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], 5])
    # Load bounding box annotations from the YOLO format
    # You'll need to parse the YOLO annotation file to get bounding boxes and class IDs
    # Assuming YOLO annotation format as: "<class_id> <center_x> <center_y> <width> <height>"
    with open(path, 'r') as file:
        lines = file.readlines()

    max_anns = min(CONFIG['MAX_BB_PER_IMAGE'], len(lines))
    for i, line in enumerate(lines):
        line = line.strip().split()
        class_id = int(line[0])
        bbox = list(map(float, line[1:]))
        bboxes[i, :4] = bbox
        bboxes[i, 4] = class_id

    bboxes[max_anns:, 4] = CONFIG['BACKGROUND_LABEL']
    return bboxes
