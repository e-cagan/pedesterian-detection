import pytest
import numpy as np
from src.utils import iou, nms

def test_iou():
    box1 = (50, 50, 150, 150)    # (x1, y1, x2, y2) coordinates of the first bounding box
    box2 = (100, 100, 200, 200)  # (x1, y1, x2, y2) coordinates of the second bounding box

    assert np.isclose(iou(box1, box2), 0.14285714285714285)


def test_iou_zero_division_error():
    box1 = (0, 0, 0, 0)
    box2 = (1, 1, 1, 1)

    assert np.isclose(iou(box1, box2), 0.0)


def test_nms():
    boxes = [
        (10, 10, 50, 50, 0.9),      # 1st box: Highest score, will be selected.
        (12, 12, 48, 48, 0.8),      # 2nd box: Will be suppressed due to high IoU with the 1st box.
        (100, 100, 200, 200, 0.75), # 3rd box: Different region, low IoU and score above threshold, will be kept.
        (20, 20, 40, 40, 0.4)       # 4th box: Will be filtered out initially because score is below conf_threshold (0.7).
    ]

    assert nms(boxes, conf_threshold=0.7, iou_threshold=0.4) == [(10, 10, 50, 50, 0.9), (100, 100, 200, 200, 0.75)]