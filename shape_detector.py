"""
shape_detector.py
-----------------
Analyses the ink on the canvas and attempts to recognise simple shapes:
  Circle, Rectangle, Triangle, Line, or None.

Detection pipeline
------------------
1. Extract a grayscale version of the alpha channel (ink mask).
2. Find contours.
3. For each significant contour apply geometry tests.
4. Return the most recently detected shape name + its bounding box.
"""

import cv2
import numpy as np
import math


class ShapeDetector:
    """
    Stateful shape detector.  Call `detect(canvas)` each frame;
    it returns (shape_name | None, label_position | None).
    """

    def __init__(self, min_contour_area: int = 1500):
        self.min_area          = min_contour_area
        self.last_shape        = None
        self.last_label_pos    = None
        self._stable_count     = 0     # consecutive frames with same shape
        self._stable_threshold = 8     # frames before we announce

    # ------------------------------------------------------------------

    def detect(self, canvas: np.ndarray):
        """
        canvas : BGRA ndarray  (the drawing canvas).
        Returns (shape_name_or_None, (label_x, label_y)_or_None).
        """
        # Use the alpha channel as an ink mask
        alpha = canvas[:, :, 3]
        if alpha.max() == 0:
            self._reset()
            return None, None

        # Light blur → binary threshold → find contours
        blurred = cv2.GaussianBlur(alpha, (5, 5), 1)
        _, binary = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            self._reset()
            return None, None

        # Work on the largest contour only
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_area:
            self._reset()
            return None, None

        shape = self._classify(largest, area)
        x, y, w, h = cv2.boundingRect(largest)
        label_pos = (x + w // 2, y - 10)

        if shape == self.last_shape:
            self._stable_count += 1
        else:
            self.last_shape     = shape
            self._stable_count  = 1
            self.last_label_pos = label_pos

        if self._stable_count >= self._stable_threshold:
            return self.last_shape, self.last_label_pos

        return None, None

    # ------------------------------------------------------------------

    def _classify(self, contour, area: float) -> str:
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "Unknown"

        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)

        # --- Circularity test ---
        circularity = (4 * math.pi * area) / (perimeter ** 2)

        if circularity > 0.75:
            return "Circle"

        # --- Rectangle / Square ---
        if vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h) if h != 0 else 1.0
            if 0.7 <= aspect <= 1.3:
                return "Square"
            return "Rectangle"

        # --- Triangle ---
        if vertices == 3:
            return "Triangle"

        # --- Line (very elongated bounding box) ---
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            return "Line"
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 6 and vertices <= 6:
            return "Line"

        if vertices >= 8:
            return "Circle"

        return "Polygon"

    # ------------------------------------------------------------------

    def _reset(self):
        self.last_shape     = None
        self.last_label_pos = None
        self._stable_count  = 0

    # ------------------------------------------------------------------

    def draw_label(self, frame: np.ndarray, shape: str, pos: tuple) -> None:
        """Render the detected shape name on *frame*."""
        if not shape or not pos:
            return
        x, y = pos
        y = max(y, 30)   # don't go off the top

        label = f"Shape: {shape}"

        # Shadow
        cv2.putText(
            frame, label, (x + 2, y + 2),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA,
        )
        # Foreground
        cv2.putText(
            frame, label, (x, y),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 255, 180), 2, cv2.LINE_AA,
        )
