"""
main.py
-------
Air Drawing Whiteboard – entry point.

Run:
    python main.py

Keyboard shortcuts:
    Q / ESC   – quit
    C         – clear canvas
    Z         – undo
    S         – save drawing as PNG
    F         – toggle full-screen
"""

import sys
import time
import datetime

import cv2
import numpy as np

from hand_tracker  import HandTracker
from drawing_utils import DrawingManager, TOOLBAR_H
from shape_detector import ShapeDetector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEBCAM_INDEX     = 0
FRAME_W          = 1280
FRAME_H          = 720
FLIP_HORIZONTAL  = True    # mirror the feed so it feels natural

# How long (seconds) to display status messages
STATUS_DURATION  = 2.0

# Shape-detection is a heavier pass; run it every N frames
SHAPE_DETECT_INTERVAL = 6


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

GESTURE_LABELS = {
    "draw":    ("DRAW",    (0, 220, 100)),
    "select":  ("SELECT",  (220, 180, 0)),
    "erase":   ("ERASE",   (0, 100, 255)),
    "fist":    ("PAUSED",  (100, 100, 100)),
    "unknown": ("???",     (180, 180, 180)),
    "none":    ("NO HAND", (60,  60,  60)),
}


def put_text_with_bg(
    frame, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.55, color=(255, 255, 255), thickness=1,
    bg_color=(40, 40, 40), padding=5,
):
    """Draw text with a solid background rectangle for readability."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color, -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hud(
    frame, gesture: str, fps: float,
    status_msg: str, status_color,
    shape_name, shape_pos,
    shape_detector: ShapeDetector,
    drawing: DrawingManager,
):
    """Render all on-screen HUD elements (below toolbar)."""

    h, w = frame.shape[:2]
    right_x = w - 260

    # ── Gesture indicator ──────────────────────────────────────────────
    label, color = GESTURE_LABELS.get(gesture, ("???", (180, 180, 180)))
    put_text_with_bg(frame, f"Mode: {label}", (right_x, TOOLBAR_H + 35),
                     scale=0.65, color=color, bg_color=(25, 25, 25))

    # ── FPS counter ────────────────────────────────────────────────────
    put_text_with_bg(frame, f"FPS: {fps:.0f}", (right_x, TOOLBAR_H + 70),
                     scale=0.55, color=(160, 220, 160), bg_color=(25, 25, 25))

    # ── Active colour swatch ───────────────────────────────────────────
    swatch_x, swatch_y = right_x, TOOLBAR_H + 100
    cv2.rectangle(frame,
                  (swatch_x, swatch_y),
                  (swatch_x + 120, swatch_y + 28),
                  drawing.color, -1)
    cv2.rectangle(frame,
                  (swatch_x, swatch_y),
                  (swatch_x + 120, swatch_y + 28),
                  (200, 200, 200), 1)
    put_text_with_bg(frame, "Colour",
                     (swatch_x, swatch_y - 4),
                     scale=0.4, color=(200, 200, 200), bg_color=(25, 25, 25))

    # ── Brush size indicator ───────────────────────────────────────────
    bs = drawing.brush_size
    cv2.circle(frame,
               (right_x + 20, TOOLBAR_H + 155),
               min(bs, 20), drawing.color, -1)
    put_text_with_bg(frame, f"Brush: {bs}px",
                     (right_x + 45, TOOLBAR_H + 160),
                     scale=0.45, color=(200, 200, 200), bg_color=(25, 25, 25))

    # ── Status message (temporary) ─────────────────────────────────────
    if status_msg:
        cy = h - 40
        put_text_with_bg(frame, status_msg, (20, cy),
                         scale=0.7, color=status_color,
                         bg_color=(20, 20, 20), padding=8)

    # ── Keyboard hints (bottom right) ─────────────────────────────────
    hints = ["Q/ESC: Quit", "C: Clear", "Z: Undo", "S: Save", "F: Full-screen"]
    for i, hint in enumerate(hints):
        put_text_with_bg(frame, hint,
                         (w - 180, h - 130 + i * 22),
                         scale=0.35, color=(140, 140, 140),
                         bg_color=(20, 20, 20), padding=2)

    # ── Shape label (drawn by ShapeDetector) ──────────────────────────
    if shape_name and shape_pos:
        shape_detector.draw_label(frame, shape_name, shape_pos)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Camera setup ---------------------------------------------------
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam (index {WEBCAM_INDEX}).")
        print("  • Check that a webcam is connected.")
        print("  • Try changing WEBCAM_INDEX in main.py.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read one frame to get actual dimensions (may differ from requested)
    ret, sample = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame from webcam.")
        sys.exit(1)
    actual_h, actual_w = sample.shape[:2]

    # --- Module init ----------------------------------------------------
    tracker  = HandTracker()
    drawing  = DrawingManager(actual_w, actual_h)
    shape_det = ShapeDetector()

    # --- Window ---------------------------------------------------------
    WIN_NAME = "Air Drawing Whiteboard"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, actual_w, actual_h)

    fullscreen   = False
    prev_time    = time.time()
    frame_count  = 0

    status_msg   = "Ready! Point your index finger to draw."
    status_color = (100, 220, 255)
    status_until = time.time() + STATUS_DURATION * 2

    detected_shape     = None
    detected_shape_pos = None

    prev_gesture  = "none"
    last_save_btn_time = 0   # debounce toolbar save

    print("[INFO] Air Drawing Whiteboard started.")
    print("  Gesture guide:")
    print("    Index finger        → Draw")
    print("    Index + Middle      → Select / Hover toolbar")
    print("    Index+Middle+Ring   → Erase")
    print("    Fist                → Pause drawing")
    print()

    while True:
        # ── Capture frame ────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Dropped frame.")
            continue

        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        # ── FPS ──────────────────────────────────────────────────────────
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        frame_count += 1

        # ── Hand tracking ────────────────────────────────────────────────
        hand_found = tracker.process(frame)
        tracker.draw_landmarks(frame)

        gesture    = tracker.detect_gesture()
        index_tip  = tracker.get_index_tip()

        # ── Gesture logic ─────────────────────────────────────────────────
        if gesture != prev_gesture:
            drawing.stop_stroke()   # end any in-progress stroke

        hover_btn = None

        if hand_found and index_tip:
            ix, iy = index_tip

            if gesture == "select":
                # Hover / click toolbar buttons
                hover_btn = drawing.toolbar_hit(ix, iy)
                if hover_btn:
                    action = drawing.activate_button(hover_btn)
                    if action == "save":
                        # Debounce: don't spam-save
                        if now - last_save_btn_time > 2.0:
                            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            path = drawing.save_png(f"drawing_{ts}.png")
                            status_msg   = f"Saved: {path}"
                            status_color = (100, 255, 180)
                            status_until = now + STATUS_DURATION
                            last_save_btn_time = now
                    elif action in ("clear", "undo"):
                        status_msg   = action.capitalize() + " ✓"
                        status_color = (100, 220, 255)
                        status_until = now + STATUS_DURATION

            elif gesture == "draw":
                drawing.save_state()
                drawing.draw_point(ix, iy)

            elif gesture == "erase":
                drawing.erase(ix, iy)
                # Show eraser circle on frame
                cv2.circle(frame, (ix, iy), 40, (150, 150, 150), 2)

            elif gesture in ("fist", "unknown", "none"):
                drawing.stop_stroke()

        else:
            drawing.stop_stroke()

        prev_gesture = gesture

        # ── Shape detection (every N frames) ─────────────────────────────
        if frame_count % SHAPE_DETECT_INTERVAL == 0:
            detected_shape, detected_shape_pos = shape_det.detect(drawing.canvas)

        # ── Composite canvas over frame ───────────────────────────────────
        composited = drawing.composite(frame)

        # ── Toolbar ───────────────────────────────────────────────────────
        drawing.draw_toolbar(composited, hover_btn=hover_btn)

        # ── HUD ───────────────────────────────────────────────────────────
        current_status = status_msg if now < status_until else ""
        draw_hud(
            composited, gesture, fps,
            current_status, status_color,
            detected_shape, detected_shape_pos,
            shape_det, drawing,
        )

        # ── Index finger cursor dot ───────────────────────────────────────
        if hand_found and index_tip and gesture in ("draw", "select", "erase"):
            ix, iy = index_tip
            dot_col = (0, 255, 120) if gesture == "draw" else \
                      (0, 180, 255) if gesture == "select" else (0, 80, 255)
            cv2.circle(composited, (ix, iy), 8, dot_col, -1)
            cv2.circle(composited, (ix, iy), 8, (255, 255, 255), 1)

        # ── Render ────────────────────────────────────────────────────────
        cv2.imshow(WIN_NAME, composited)

        # ── Keyboard handling ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):   # Q or ESC
            break

        elif key == ord('c'):
            drawing.clear()
            status_msg   = "Canvas cleared"
            status_color = (100, 220, 255)
            status_until = now + STATUS_DURATION

        elif key == ord('z'):
            drawing.undo()
            status_msg   = "Undo"
            status_color = (100, 220, 255)
            status_until = now + STATUS_DURATION

        elif key == ord('s'):
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = drawing.save_png(f"drawing_{ts}.png")
            status_msg   = f"Saved → {path}"
            status_color = (100, 255, 180)
            status_until = now + STATUS_DURATION
            print(f"[INFO] Drawing saved: {path}")

        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("[INFO] Shutting down.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
