#bbox1 = [3, 4, 6, 7]
#bbox2 = [3, 4, 6, 7]

def iou(bbox1, bbox2):

    bbox1_area = abs(bbox1[0] - bbox1[2]) * abs(bbox1[1] - bbox1[3])
    bbox2_area = abs(bbox2[0] - bbox2[2]) * abs(bbox2[1] - bbox2[3])
    bbox_intersection = (max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3]))
    intersection_area = (bbox_intersection[2] - bbox_intersection[0]) * (bbox_intersection[3] - bbox_intersection[1])

    if intersection_area < 0:
        intersection_area = 0
    iou_ = intersection_area / (bbox1_area + bbox2_area - intersection_area + 1e-6)


    #print('bbox1_area: ', bbox1_area)
    #print('bbox2_area: ', bbox2_area)
    #print('bbox_intersection: ', bbox_intersection)
    #print('intersection_area: ', intersection_area)
    #print('iou: {:.2f}'.format(iou))
    
    return iou_, bbox_intersection
#iou(bbox1, bbox2)
