from typing import Tuple, List, Union
import tensorflow as tf
from code_loader.helpers.detection.yolo.utils import reshape_output_list

from coco_od.config import CONFIG
from coco_od.yolo_helpers.yolo_utils import LOSS_FN


def compute_losses(obj_true: tf.Tensor, od_pred: tf.Tensor) -> Union[
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]],
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    decoded = False if CONFIG["MODEL_FORMAT"] != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(od_pred, decoded=decoded,
                                                                 image_size=CONFIG["IMAGE_SIZE"])  # add batch
    loss_l, loss_c, loss_o = LOSS_FN(y_true=obj_true, y_pred=(loc_list_reshaped, class_list_reshaped))
    return loss_l, loss_c, loss_o


def od_loss(bb_gt: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:  # return batch
    """
    Sums the classification and regression loss
    """
    loss_l, loss_c, loss_o = compute_losses(bb_gt, y_pred)
    combined_losses = [l + c + o for l, c, o in zip(loss_l, loss_c, loss_o)]
    sum_loss = tf.reduce_sum(combined_losses, axis=0)
    non_nan_loss = tf.where(tf.math.is_nan(sum_loss), tf.zeros_like(sum_loss), sum_loss)  # LOSS 0 for NAN losses
    return non_nan_loss



