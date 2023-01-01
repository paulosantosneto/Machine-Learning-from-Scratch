from intersection_over_union import iou

def nms(bboxes: list, prob_threshold: float, iou_threshold: float) -> list:
    """
    @param bboxes: predicted bounding boxes.
    @param prob_threshold: accepted minimum probability limit.
    @param iou_threshold: accepted minimum iou limit.
    @returns: list with a single bounding box for each class.
    @raises AssertionError: if bbox is not a list.
    """

    # predictions : [class, probability, x1, y1, x2, y2]

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes_after_nms = []
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes if box[0] != chosen_box[0] and 
            iou(chosen_box[2:], box[2:]) < iou_threshold]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
