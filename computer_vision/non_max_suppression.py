from intersection_over_union import iou

def nms(bboxes: list, prob_threshold: float, iou_threshold: float) -> list:
    """
    @param bboxes: predicted bounding boxes.
    @param prob_threshold: accepted minimum probability limit.
    @param iou_threshold: accepted minimum iou limit.
    @returns: list with a single bounding box for each class.
    @raises AssertionError: if bbox is not a list.
    """

    # predictions : [image_idx, class, probability, color, x1, y1, x2, y2]

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[2] > prob_threshold]
    bboxes_after_nms = []
    bboxes = sorted(bboxes, key=lambda x: x[2], reverse=True)

    while bboxes:
        chosen_box = bboxes.pop(0)

        aux = sorted([box for box in bboxes if (box[1] == chosen_box[1] and iou(chosen_box[4:], box[4:])[0] > 0)], key=lambda x: x[2], reverse=True)

        bboxes_after_nms.append(chosen_box)
        
        for box in aux:
            bboxes.remove(box)

    return bboxes_after_nms
