
def iou(bbox1, bbox2):
    """
    This function calculate intersection over union between two bounding boxes.

    @param bbox1: first bounding box [x1, y1, x2, y2].
    @param bbox2: second bounding box [x1, y1, x2, y2].
    @returns: (value iou, bounding box intersection).
    @raises AssertionError: it's not a list.
    """
    
    assert bbox1 == list
    assert bbox2 == list

    bbox1_area = abs(bbox1[0] - bbox1[2]) * abs(bbox1[1] - bbox1[3])
    bbox2_area = abs(bbox2[0] - bbox2[2]) * abs(bbox2[1] - bbox2[3])
    
    # upper left corner:
    #          x : max(bbox1[0], bbox2[0])
    #          y : max(bbox1[1], bbox2[1])
    # bottom right corner:
    #          x : min(bbox1[0], bbox2[0])
    #          y : min(bbox1[1], bbox2[1])
    bbox_intersection = (max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3]))
    intersection_area = (bbox_intersection[2] - bbox_intersection[0]) * (bbox_intersection[3] - bbox_intersection[1])

    if intersection_area < 0:
        intersection_area = 0
    iou_ = intersection_area / (bbox1_area + bbox2_area - intersection_area + 1e-6)

    return iou_, bbox_intersection

