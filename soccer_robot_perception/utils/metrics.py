import typing
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance


def calculate_metrics(
    target: np.ndarray,
    output: np.ndarray,
    include_bg_class: bool = False,
    average: str = None,
    num_classes: int = 3,
) -> typing.Tuple:
    """
    Calculates the evaluation metrics precision, recall and f-score for the average
    method passed using sklearn.metrics.precision_recall_fscore_support.

    Args:
        target: np.ndarray
        Flattened label prediction
        output: np.ndarray
        Flattened model prediction
        include_bg_class: bool
        Flag to whether include background class in metric calculation
        average: string
        Average argument for sklearn.metrics.precision_recall_fscore_support

    Returns:
        precision, recall, fscore: tuple
        Evaluation metrics: precision, recall and fscore respectively

    """
    if include_bg_class:
        start_label = 0
    else:
        start_label = 1
    metrics = precision_recall_fscore_support(
        target[0],
        output[0],
        labels=list(range(start_label, num_classes)),
        average=average,
    )
    accuracy = accuracy_score(target[0], output[0])

    return metrics[0], metrics[1], metrics[2], accuracy


def get_confusion_matrix(
    target: np.ndarray,
    output: np.ndarray,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Calculates the confusion matrix normalized for each image with regard to all the character classes
    method passed using sklearn.metrics.confusion_matrix.

    Args:
        target: np.ndarray
        Flattened label prediction
        output: np.ndarray
        Flattened model prediction

    Returns:
        confusion_matrix_: np.ndarray
        Confusion matrix for all the characters

    """

    confusion_matrix_ = confusion_matrix(
        target[0],
        output[0],
        labels=list(range(0, num_classes)),
        normalize=None,
    )
    return confusion_matrix_


def iou_metrics_preprocess(tensor_input):

    bg_iou_map = tensor_input.clone()
    field_iou_map = tensor_input.clone()
    lines_iou_map = tensor_input.clone()
    bg_iou_map[bg_iou_map > 0] = 10
    bg_iou_map[bg_iou_map == 0] = 1
    bg_iou_map[bg_iou_map == 10] = 0
    field_iou_map[field_iou_map != 1] = 0
    lines_iou_map[lines_iou_map < 2] = 0
    lines_iou_map[lines_iou_map == 2] = 1

    return bg_iou_map, field_iou_map, lines_iou_map


def calculate_iou(target_img, input_img):

    input_img = input_img.squeeze(0)
    target_img = target_img.squeeze(0)

    input_img = input_img.int()
    target_img = target_img.int()

    intersection_image = (input_img & target_img).float()

    intersection_pixels = intersection_image.sum(0).sum(0)
    union_image = (input_img | target_img).float()
    union_pixels = union_image.sum(0).sum(0)

    return torch.clamp(
        (intersection_pixels / torch.clamp(union_pixels, min=1e-5)),
        min=0.0,
        max=1.0,
    )


def calculate_det_metrics_old(predicted_points, gt_blob_centers, name):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    gt_blob_centers = gt_blob_centers.numpy()
    gather_idx = np.where(gt_blob_centers[:, 2] == name)[0]
    gather_data = []
    for i in gather_idx:
        gather_data.append(list(gt_blob_centers[i]))

    flag = np.zeros((len(gather_data)))
    for i in predicted_points:
        count = -1
        for j in gather_data:
            object_detected = False
            if i[0] > 0 and i[1] > 0:
                object_detected = True

            if object_detected:
                distance_ = distance.euclidean(j, i)
                if distance_ < 4:
                    tp += 1
                    count = 1
                    break
        if count == -1:
            fp += 1
        else:
            flag[count] = 1

    if len(predicted_points) == 0 and (not np.any(np.array(gather_data))):
        tn += 1
    fn = np.count_nonzero(flag)

    print("Metrics: ", tp, tn, fp, fn)
    print("Predicted point: ", predicted_points)
    print("Gathered point: ", gather_data)
    print("Haha")

    return 0, 0, 0, 0, 0


def calculate_det_metrics(predicted_points, gt_blob_centers, name):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    gt_blob_centers = gt_blob_centers.numpy()
    gather_data = gt_blob_centers[gt_blob_centers[:, 2] == name]
    actual = np.zeros((len(gather_data)))
    predicted = np.ones((len(predicted_points)))

    for n, i in enumerate(predicted_points):
        for m, j in enumerate(gather_data):
            object_detected = False
            if i[0] > 0 or i[1] > 0:
                object_detected = True

            if object_detected:
                distance_ = distance.euclidean(j, i)
                if distance_ < 4:
                    tp += 1
                    actual[m] = 1
                    break

        fp = len(predicted) - tp

    if len(predicted_points) == 0 and (not np.any(np.array(gather_data))):
        tn += 1

    fn = np.count_nonzero(actual == 0)

    precision = calculate_det_precision(tp, fp, tn, fn)
    recall = calculate_det_recall(tp, fp, tn, fn)
    f1 = calculate_det_f1(precision, recall)
    accuracy = calculate_det_accuracy(tp, fp, tn, fn)
    fdr = calculate_det_fdr(tp, fp, tn, fn)

    return tp, fp, tn, fn, precision, recall, f1, accuracy, fdr


def calculate_det_precision(tp, fp, tn, fn):
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    return precision


def calculate_det_recall(tp, fp, tn, fn):
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    return recall


def calculate_det_f1(precision, recall):
    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return f1


def calculate_det_accuracy(tp, fp, tn, fn):
    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        accuracy = 0
    return accuracy


def calculate_det_fdr(tp, fp, tn, fn):
    try:
        fdr = fp / (fp + tp)
    except ZeroDivisionError:
        fdr = 0
    return fdr
