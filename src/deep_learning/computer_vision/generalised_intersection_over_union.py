from intersection_over_union import iou

def giou(bbox1: list, bbox2: list) -> float:
    """
    This function calculate generalised intersection over union between two bounding boxes.

    @param bbox1: a list of the form: [x1, y1, x2, y2]
    @param bbox2: a list of the form: [x1, y1, x2, y2]
    @returns: the value of the giou metric.
    @raises: assert error when bounding boxes are not in accepted form.
    """

    assert len(bbox1) == len(bbox2)
    
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) # or A in the equation
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) # or B in the equation

    iou_value, intersection = iou(bbox1, bbox2)

    intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])

    x_left, y_left, x_right, y_right = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),
                                        max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]

    maximized_area = (x_right - x_left) * (y_right - y_left) # or C in the equation
    
    # iou - ((c \ |A U B|) / |C|)
    giou_value = iou_value - ((maximized_area - (area_bbox1 + area_bbox2 - intersection_area)) / maximized_area)

    return giou_value

bbox1 = [10, 15, 20, 30]
bbox2 = [10, 15, 20, 30]
print(giou(bbox1, bbox2))
