import pytest
import numpy as np
from src.utils import iou

def test_iou():
    box1 = (50, 50, 150, 150)    # (x1, y1, x2, y2) coordinates of the first bounding box
    box2 = (100, 100, 200, 200)  # (x1, y1, x2, y2) coordinates of the second bounding box

    assert np.isclose(iou(box1, box2), 0.14285714285714285)