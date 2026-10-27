import numpy as np

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

    area_of_intersection = max(0, (x6 - x5)) * max(0, (y6 - y5))

    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)

    area_of_union = area_box1 + area_box2 - area_of_intersection

    try:
        return area_of_intersection / area_of_union
    except ZeroDivisionError:
        return 0.0


def nms(boxes, conf_threshold=0.7, iou_threshold=0.4):
    """
    A box is expected in this shape: (x1, y1, x2, y2, conf_score)
    """
    
    bbox_list_thresholded = []
    bbox_list_new = []

    bbox_sorted = sorted(boxes, reverse=True, key=lambda x: x[4])
    for box in bbox_sorted:
        if box[4] > conf_threshold:
            bbox_list_thresholded.append(box)

    while len(bbox_list_thresholded) > 0:
        current_box = bbox_list_thresholded.pop(0)
        bbox_list_new.append(current_box)

        for box in bbox_list_thresholded.copy():
            _iou = iou(current_box[:4], box[:4])
            if _iou > iou_threshold:
                bbox_list_thresholded.remove(box)

    return bbox_list_new