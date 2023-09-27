import onnxruntime as ort
import numpy as np
import tensorflow as tf

from coco_od.metrics import od_loss
from coco_od.visualizers.visualizers import (
    gt_bb_decoder, bb_decoder
)
from leap_binder import (
    subset_images, input_image, get_bbs, general_metrics_dict,
    metadata_dict,
)


def check_integration():
    model_path = 'model/yolov5s.h5'
    model = tf.keras.models.load_model(model_path)
    batch = 16
    responses = subset_images()  # get dataset splits
    training_response = responses[0]  # [training, validation, test]
    images = []
    bb_gt = []
    for idx in range(batch):
        images.append(input_image(idx, training_response))
        bb_gt.append(get_bbs(idx, training_response))
    y_true_bbs = tf.convert_to_tensor(bb_gt)  # convert ground truth bbs to tensor

    input_img_tf = tf.convert_to_tensor(images)
    y_pred_bbs = model([input_img_tf])  # infer and get model prediction
    y_pred_bb_concat = tf.keras.layers.Permute((2, 1))(y_pred_bbs)  # prepare prediction for further use

    # loss
    loss = od_loss(y_true_bbs, y_pred_bb_concat)
    # custom metrics
    general_metric_results = general_metrics_dict(y_true_bbs, y_pred_bb_concat)

    # visualizers
    predicted_bboxes_img = bb_decoder(images[0], y_pred_bb_concat[0, ...])
    gt_bboxes_img = gt_bb_decoder(images[0], y_pred_bb_concat[0, ...])

    metadata = metadata_dict(idx, training_response)


if __name__ == '__main__':
    check_integration()
