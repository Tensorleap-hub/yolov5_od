import os
from typing import Union, List, Dict
import numpy as np
import tensorflow as tf
from PIL import Image

from code_loader import leap_binder
from code_loader.contract.enums import LeapDataType

from code_loader.contract.datasetclasses import PreprocessResponse

from yolo_od.config import CONFIG, dataset_path
from yolo_od.data.preprocessing import load_yolo_dataset
from yolo_od.utils.general_utils import extract_bboxes_yolo
from yolo_od.metrics import compute_losses, od_loss
from yolo_od.visualizers.visualizers import gt_bb_decoder, bb_decoder


# ----------------------------------------------------data processing--------------------------------------------------
def subset_images() -> List[PreprocessResponse]:
    """
    This function returns the training and validation datasets in the format expected by tensorleap
    """
    train = load_yolo_dataset(dataset_path, split='train')
    val = load_yolo_dataset(dataset_path, split='val')
    test = load_yolo_dataset(dataset_path, split='test')

    train_size = min(len(train), CONFIG['TRAIN_SIZE'])
    val_size = min(len(val), CONFIG['VAL_SIZE'])
    test_size = min(len(test), CONFIG['TEST_SIZE'])

    np.random.seed(0)
    train_idx, val_idx, test_idx = (np.random.choice(len(train), train_size, replace=False),
                                    np.random.choice(len(val), val_size, replace=False),
                                    np.random.choice(len(test), test_size, replace=False))
    training_subset = PreprocessResponse(length=train_size, data={'files': train,
                                                                  'samples': np.take(train, train_idx),
                                                                  'subdir': 'train'})
    validation_subset = PreprocessResponse(length=val_size, data={'files': val,
                                                                  'samples': np.take(val, val_idx),
                                                                  'subdir': 'val'})
    test_subset = PreprocessResponse(length=test_size, data={'files': test,
                                                             'samples': np.take(test, test_idx),
                                                             'subdir': 'test'})
    return [training_subset, validation_subset, test_subset]


def input_image(idx: int, data: PreprocessResponse, scale: bool = True) -> np.ndarray:
    """
    Returns a BGR image normalized and padded
    """
    data = data.data
    image_file_name = data['samples'][idx] + '.jpg'
    path = os.path.join(dataset_path, f"{data['subdir']}/images/{image_file_name}")

    # rescale
    if scale:
        image = np.array(
            Image.open(path).resize((CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1]), Image.BILINEAR)) / 255.
    else:
        image = np.array(Image.open(path))
    return image


def get_bbs(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    labels_file_name = data['samples'][idx] + '.txt'
    path = os.path.join(dataset_path, f"{data['subdir']}/labels/{labels_file_name}")
    bboxes = extract_bboxes_yolo(path)
    return bboxes


# ----------------------------------------------------------metadata----------------------------------------------------
def get_fname(index: int, subset: PreprocessResponse) -> str:
    data = subset.data
    fname = data['samples'][index]
    return fname


def get_original_width(img: np.ndarray) -> int:
    return img.shape[1]


def get_original_height(img: np.ndarray) -> int:
    return img.shape[0]


def bbox_num(bbs: np.ndarray) -> int:
    number_of_bb = np.count_nonzero(bbs[..., -1] != CONFIG['BACKGROUND_LABEL'])
    return number_of_bb


def get_avg_bb_area(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    return areas.mean()


def get_avg_bb_aspect_ratio(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()


def get_instances_num(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    return float(valid_bbs.shape[0])


# def get_object_instances_num(bbs: np.ndarray) -> float:
#     label = CONFIG['CATEGORIES'].index('Object')
#     valid_bbs = bbs[bbs[..., -1] == label]
#     return float(valid_bbs.shape[0])


# def get_obj_bbox_occlusions_count(img: np.ndarray, bboxes: np.ndarray, calc_avg_flag=False) -> float:
#     occlusion_threshold = 0.2  # Example threshold value
#     occlusions_count = count_obj_bbox_occlusions(img, bboxes, occlusion_threshold, calc_avg_flag)
#     return occlusions_count


# def get_obj_bbox_occlusions_avg(img: np.ndarray, bboxes: np.ndarray) -> float:
#     return get_obj_bbox_occlusions_count(img, bboxes, calc_avg_flag=True)


def count_duplicate_bbs(bbs_gt: np.ndarray) -> int:
    real_gt = bbs_gt[bbs_gt[..., 4] != CONFIG['BACKGROUND_LABEL']]
    return int(real_gt.shape[0] != np.unique(real_gt, axis=0).shape[0])


def count_small_bbs(bboxes: np.ndarray) -> float:
    obj_boxes = bboxes[bboxes[..., -1] == 0]
    areas = obj_boxes[..., 2] * obj_boxes[..., 3]
    return float(len(areas[areas < CONFIG['SMALL_BBS_TH']]))


def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    bbs = get_bbs(idx, data)
    img = input_image(idx, data, scale=False)

    metadatas = {
        "idx": idx,
        "fname": get_fname(idx, data),
        "origin_width": get_original_width(img),
        "origin_height": get_original_height(img),
        "instances_number": get_instances_num(bbs),
        # "object_number": get_object_instances_num(bbs),
        "bbox_number": bbox_num(bbs),
        "bbox_area": get_avg_bb_area(bbs),
        "bbox_aspect_ratio": get_avg_bb_aspect_ratio(bbs),
        "duplicate_bb": count_duplicate_bbs(bbs),
        "small_bbs_number": count_small_bbs(bbs),
        # "count_total_obj_bbox_occlusions": get_obj_bbox_occlusions_count(img, bbs),
        # "avg_obj_bbox_occlusions": get_obj_bbox_occlusions_avg(img, bbs),
    }

    return metadatas


def general_metrics_dict(bb_gt: tf.Tensor, detection_pred: tf.Tensor) -> Dict[str, tf.Tensor]:
    try:
        reg_met, class_met, obj_met = compute_losses(bb_gt, detection_pred)
    except Exception as e:
        print(e)
        batch_dim = bb_gt.shape[0]
        fault_res_tensor = [tf.convert_to_tensor(-np.ones((batch_dim, 1))) for _ in range(3)]
        reg_met, class_met, obj_met = (fault_res_tensor, fault_res_tensor, fault_res_tensor, fault_res_tensor)
    res = {
        "Regression_metric": tf.reduce_sum(reg_met, axis=0)[:, 0],
        "Classification_metric": tf.reduce_sum(class_met, axis=0)[:, 0],
        "Objectness_metric": tf.reduce_sum(obj_met, axis=0)[:, 0],
    }
    return res


# ---------------------------------------------------------binding------------------------------------------------------
# preprocess function
leap_binder.set_preprocess(subset_images)

# unlabeled data preprocess
# set input and gt
leap_binder.set_input(input_image, 'images')
leap_binder.set_ground_truth(get_bbs, 'bbs')
# set prediction (object)
leap_binder.add_prediction('object detection',
                           ["x", "y", "w", "h", "obj"] +
                           [f"class_{i}" for i in range(CONFIG['CLASSES'])] +
                           [f"mask_coeff_{i}" for i in range(32)])

# set custom loss
leap_binder.add_custom_loss(od_loss, 'od_loss')
# set visualizers
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)

# set custom metrics
leap_binder.add_custom_metric(general_metrics_dict, 'general_metrics')

# set metadata
leap_binder.set_metadata(metadata_dict, name='metadata')

if __name__ == '__main__':
    leap_binder.check()
