"""
drawing_utils.py
----------------
Manages the virtual canvas, stroke history (undo), brush/color state,
toolbar rendering and hit-testing, and canvas-to-frame compositing.
"""

import cv2
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOLBAR_H = 80          # height of the top toolbar in pixels

# Colour palette  (BGR)
COLORS = {
    "Red":    (0,   0,   220),
    "Green":  (0,   200, 50),
    "Blue":   (220, 80,  0),
    "Yellow": (0,   220, 220),
    "Black":  (10,  10,  10),
}

BRUSH_SIZES = {"Small": 4, "Medium": 10, "Large": 22}
ERASER_SIZE = 40


class DrawingManager:
    """
    Owns the transparent drawing canvas and all drawing state.
    """

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        # BGRA canvas (A=0 → transparent, A=255 → opaque ink)
        self.canvas: np.ndarray = np.zeros((height, width, 4), dtype=np.uint8)

        # Stroke history for undo  (each entry = canvas snapshot as BGRA)
        self._history: deque = deque(maxlen=20)

        # Current drawing state
        self.color      = list(COLORS.values())[0]   # default Red (BGR)
        self.brush_size = BRUSH_SIZES["Medium"]
        self.prev_point = None                        # last drawn point

        # Smoothing buffer
        self._smooth_x: deque = deque(maxlen=5)
        self._smooth_y: deque = deque(maxlen=5)

        # Toolbar button layout (built lazily once we know width)
        self._toolbar_buttons: list[dict] = []
        self._build_toolbar(width)

    # ------------------------------------------------------------------
    # Toolbar layout
    # ------------------------------------------------------------------

    def _build_toolbar(self, w: int) -> None:
        """
        Lay out toolbar buttons:
          color swatches | brush sizes | Clear | Save | Undo
        """
        self._toolbar_buttons = []
        x = 10
        pad = 8

        # --- Color buttons ---
        for name, bgr in COLORS.items():
            btn = dict(
                kind="color", label=name, value=bgr,
                x1=x, y1=5, x2=x + 55, y2=TOOLBAR_H - 10,
            )
            self._toolbar_buttons.append(btn)
            x += 60 + pad

        x += 10  # extra gap

        # --- Brush size buttons ---
        for name, sz in BRUSH_SIZES.items():
            btn = dict(
                kind="brush", label=name, value=sz,
                x1=x, y1=5, x2=x + 65, y2=TOOLBAR_H - 10,
            )
            self._toolbar_buttons.append(btn)
            x += 70 + pad

        x += 10

        # --- Action buttons ---
        for label, kind in [("Clear", "clear"), ("Undo", "undo"), ("Save", "save")]:
            btn = dict(
                kind=kind, label=label, value=None,
                x1=x, y1=5, x2=x + 60, y2=TOOLBAR_H - 10,
            )
            self._toolbar_buttons.append(btn)
            x += 65 + pad

    # ------------------------------------------------------------------
    # Coordinate smoothing
    # ------------------------------------------------------------------

    def _smooth(self, x: int, y: int) -> tuple[int, int]:
        self._smooth_x.append(x)
        self._smooth_y.append(y)
        return (
            int(np.mean(self._smooth_x)),
            int(np.mean(self._smooth_y)),
        )

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def draw_point(self, raw_x: int, raw_y: int) -> None:
        """Draw a smooth stroke segment from the previous point to (x, y)."""
        if raw_y < TOOLBAR_H:          # ignore toolbar zone
            self.prev_point = None
            return

        x, y = self._smooth(raw_x, raw_y)

        if self.prev_point is None:
            self.prev_point = (x, y)

        # Convert BGR color to BGRA for the canvas
        bgra = (*self.color, 255)

        cv2.line(
            self.canvas,
            self.prev_point,
            (x, y),
            bgra,
            self.brush_size,
            lineType=cv2.LINE_AA,
        )
        # Draw a filled circle at the endpoint for smooth round caps
        cv2.circle(self.canvas, (x, y), self.brush_size // 2, bgra, -1, cv2.LINE_AA)

        self.prev_point = (x, y)

    def erase(self, raw_x: int, raw_y: int) -> None:
        """Erase a circular region around (x, y)."""
        if raw_y < TOOLBAR_H:
            return
        x, y = self._smooth(raw_x, raw_y)
        cv2.circle(self.canvas, (x, y), ERASER_SIZE, (0, 0, 0, 0), -1)

    def stop_stroke(self) -> None:
        """Call when the user lifts the drawing finger."""
        self.prev_point = None
        self._smooth_x.clear()
        self._smooth_y.clear()

    # ------------------------------------------------------------------
    # History / undo
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Snapshot current canvas for undo."""
        self._history.append(self.canvas.copy())

    def undo(self) -> None:
        if self._history:
            self.canvas = self._history.pop()

    def clear(self) -> None:
        self.save_state()
        self.canvas[:] = 0

    # ------------------------------------------------------------------
    # Canvas → frame compositing
    # ------------------------------------------------------------------

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """
        Alpha-blend the BGRA canvas over the BGR webcam frame.
        Returns the composited BGR frame.
        """
        alpha = self.canvas[:, :, 3:4].astype(np.float32) / 255.0
        ink   = self.canvas[:, :, :3].astype(np.float32)
        bg    = frame.astype(np.float32)
        out   = (ink * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        return out

    # ------------------------------------------------------------------
    # Toolbar rendering
    # ------------------------------------------------------------------

    def draw_toolbar(self, frame: np.ndarray, hover_btn=None) -> None:
        """
        Render the toolbar bar onto *frame* (in-place).
        *hover_btn* is the button dict currently under the cursor (or None).
        """
        # Semi-transparent dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, TOOLBAR_H), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        for btn in self._toolbar_buttons:
            x1, y1, x2, y2 = btn["x1"], btn["y1"], btn["x2"], btn["y2"]
            is_hover = (btn is hover_btn)

            if btn["kind"] == "color":
                bgr = btn["value"]
                is_active = (list(bgr) == list(self.color))
                # Swatch fill
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
                border_col = (255, 255, 255) if is_active else (120, 120, 120)
                border_t   = 3 if is_active else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, border_t)
                if is_hover:
                    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

            elif btn["kind"] == "brush":
                is_active = (btn["value"] == self.brush_size)
                bg_col = (80, 80, 80) if is_active else (50, 50, 50)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bg_col, -1)
                border_col = (255, 255, 255) if is_active else (100, 100, 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, 1 if not is_active else 2)
                # Draw a sample dot showing brush size
                dot_r = min(btn["value"] // 2 + 2, 12)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 - 6
                cv2.circle(frame, (cx, cy), dot_r, (200, 200, 200), -1)
                cv2.putText(
                    frame, btn["label"],
                    (x1 + 4, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA,
                )

            else:
                # Action button
                bg_col = (70, 70, 70) if is_hover else (45, 45, 45)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bg_col, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                tw, th = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                tx = x1 + ((x2 - x1) - tw) // 2
                ty = y1 + ((y2 - y1) + th) // 2
                cv2.putText(
                    frame, btn["label"],
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
                )

    # ------------------------------------------------------------------
    # Toolbar hit-testing
    # ------------------------------------------------------------------

    def toolbar_hit(self, x: int, y: int):
        """
        Return the button dict under (x, y), or None.
        Only checks the toolbar zone.
        """
        if y > TOOLBAR_H:
            return None
        for btn in self._toolbar_buttons:
            if btn["x1"] <= x <= btn["x2"] and btn["y1"] <= y <= btn["y2"]:
                return btn
        return None

    def activate_button(self, btn: dict) -> str:
        """
        Apply button action; returns an action string for the caller.
        """
        kind = btn["kind"]
        if kind == "color":
            self.color = list(btn["value"])
            return "color"
        elif kind == "brush":
            self.brush_size = btn["value"]
            return "brush"
        elif kind == "clear":
            self.clear()
            return "clear"
        elif kind == "undo":
            self.undo()
            return "undo"
        elif kind == "save":
            return "save"
        return "none"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_png(self, path: str = "drawing.png") -> str:
        """Save current canvas merged on white background as PNG."""
        white = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        alpha = self.canvas[:, :, 3:4].astype(np.float32) / 255.0
        ink   = self.canvas[:, :, :3].astype(np.float32)
        merged = (ink * alpha + white.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        cv2.imwrite(path, merged)
        return path
