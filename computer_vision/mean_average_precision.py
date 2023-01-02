from intersection_over_union import iou
from collections import Counter
import numpy as np

def mAP(bboxes: list, labels: list, iou_threshold=0.5, prob_threshold=0.5, num_class=1) -> float:
    """
    This function implement mean average precision metric for object detection.

    @param bboxes (list): a list of all specific predicted bounding boxes of a dataset.
                          [image_idx, class, probability, x1, y1, x2, y2]
    @param labels (list): a list of all true labels bounding boxes.
                          [image_idx, class, probability, x1, y1, x2, y2]
    @param iou_threshold (float): iou metric minimum threshold.
    @param prob_threshold (float): minimum accepted probability for a bounding box.
    @param num_class (float): class number.
    @returns: value indicating mean average precision.
    @raise AssertionError: it's not a list.
    """

    assert type(bboxes) == list

    average_precision = []
    alpha = 1e-5

    for cls in range(num_class):

        cls_bboxes = [box for box in bboxes if box[1] == cls]
        cls_labels = [box for box in labels if box[1] == cls]
        
        mapping_gt = {}
        counting_gt = Counter([gt[0] for gt in cls_bboxes])

        for cont in counting_gt:
            mapping_gt[cont] = [0] * counting_gt[cont]

        cls_bboxes.sort(key=lambda x: x[2], reverse=True)
        true_positive = [0] * len(cls_bboxes)
        false_positive = [0] * len(cls_bboxes)
        total_gt = len(cls_labels)

        for box_idx, box in enumerate(cls_bboxes):
            img_gt = [gt for gt in cls_labels if gt[0] == box[0]]
            
            num_gts = len(img_gt)
            best_iou_val = 0

            for idx, gt in enumerate(img_gt):
                iou_val, _ = iou(gt[3:], box[3:])

                if iou_val > best_iou_val:
                    best_iou_val = iou_val
                    best_iou_idx = idx
            
            if best_iou_val > iou_threshold:
                if mapping_gt[box[0]][best_iou_idx] == 0:
                    true_positive[box_idx] = 1 
                    mapping_gt[box[0]][best_iou_idx] = 1
                else:
                    false_positive[box_idx] = 1
            else:
                false_positive[box_idx] = 1

            tp_sum = np.cumsum(true_positive, axis=0) 
            fp_sum = np.cumsum(false_positive, axis=0)
            precisions = np.divide(tp_sum, (tp_sum + fp_sum + alpha))
            recalls = np.divide(tp_sum, (total_gt + alpha))
            precisions = np.insert(precisions, 0, 1)
            recalls = np.insert(recalls, 0, 0)

            average_precision.append(np.trapz(precisions, recalls))

    return sum(average_precision) / len(average_precision)

