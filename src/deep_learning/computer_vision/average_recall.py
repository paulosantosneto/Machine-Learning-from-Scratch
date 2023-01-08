from .intersection_over_union import iou
import numpy as np

def AR(bboxes: list, true_labels: list) -> float:
    """
    This function implement average recall metric for objection detection.

    AR@0.5:0.05:1.00

    @param bboxes (list): a list of all specific predicted bounding boxes of a dataset.
                          [image_idx, class, probability, x1, y1, x2, y2]
    @param true_labels (list): a list of all true labels bounding boxes (ground truth).
                          [image_idx, class, probability, x1, y2, x2, y2]
    @returns: value indicating average recall
    @raise AssertionError: it's not a list.
    """

    assert type(bboxes) == list
    assert type(true_labels) == list
    
    #for cur_iou in np.arange(0.5, 1.0, 0.05):
        #print(cur_iou)
        

#predicted = [[1, 0, 0.7, 100, 50, 300, 200], [1, 0, 0.3, 350, 200, 400, 250], [2, 0, 0.8, 200, 70, 500, 320]]
#labels = [[1, 0, 1.0, 150, 35, 375, 270], [2, 0, 1.0, 190, 100, 480, 380]]
#average_recall(bboxes=predicted, true_labels=labels)
