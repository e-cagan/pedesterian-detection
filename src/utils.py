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