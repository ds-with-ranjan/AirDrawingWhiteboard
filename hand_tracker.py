"""
hand_tracker.py
---------------
Handles all MediaPipe hand detection and landmark tracking.
Provides clean abstractions for finger state detection.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """
    Wraps MediaPipe Hands for single-hand tracking with gesture detection.
    """

    # Landmark indices for fingertips and PIP joints (second knuckle)
    FINGERTIP_IDS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
    PIP_IDS        = [3, 6, 10, 14, 18]   # joints just below each tip

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.75,
        tracking_confidence: float = 0.75,
    ):
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        # Latest detection results
        self.results    = None
        self.landmarks  = None   # list of (x_px, y_px) for all 21 landmarks
        self.frame_h    = 0
        self.frame_w    = 0

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> bool:
        """
        Run detection on a BGR frame.
        Returns True if a hand was found.
        """
        self.frame_h, self.frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            hand_lm = self.results.multi_hand_landmarks[0]
            self.landmarks = [
                (
                    int(lm.x * self.frame_w),
                    int(lm.y * self.frame_h),
                )
                for lm in hand_lm.landmark
            ]
            return True

        self.landmarks = None
        return False

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """Overlay skeleton + landmark dots on *frame* (in-place)."""
        if not self.results or not self.results.multi_hand_landmarks:
            return
        for hand_lm in self.results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_lm,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

    # ------------------------------------------------------------------
    # Landmark helpers
    # ------------------------------------------------------------------

    def get_landmark(self, idx: int):
        """Return (x, y) pixel coords for landmark *idx*, or None."""
        if self.landmarks:
            return self.landmarks[idx]
        return None

    def get_index_tip(self):
        """Shortcut: index fingertip position."""
        return self.get_landmark(8)

    # ------------------------------------------------------------------
    # Finger-state helpers
    # ------------------------------------------------------------------

    def fingers_up(self) -> list[int]:
        """
        Return a list of 5 booleans (0/1) — one per finger:
        [thumb, index, middle, ring, pinky].
        1 = finger extended, 0 = finger folded.
        """
        if not self.landmarks:
            return [0, 0, 0, 0, 0]

        up = []

        # Thumb: compare x-coords (horizontal spread)
        # Works for right hand; good enough for a mirror-flipped webcam feed.
        thumb_tip = self.landmarks[4]
        thumb_ip  = self.landmarks[3]
        up.append(1 if thumb_tip[0] < thumb_ip[0] else 0)

        # Four fingers: tip y < PIP y  →  finger is up
        for tip_id, pip_id in zip(self.FINGERTIP_IDS[1:], self.PIP_IDS[1:]):
            tip = self.landmarks[tip_id]
            pip = self.landmarks[pip_id]
            up.append(1 if tip[1] < pip[1] else 0)

        return up

    # ------------------------------------------------------------------
    # High-level gesture detection
    # ------------------------------------------------------------------

    def detect_gesture(self) -> str:
        """
        Map current finger state to a named gesture string:
          "draw"      – only index finger up
          "select"    – index + middle up
          "erase"     – index + middle + ring up (three fingers)
          "fist"      – all fingers down
          "unknown"   – anything else
        """
        if not self.landmarks:
            return "none"

        f = self.fingers_up()  # [thumb, index, middle, ring, pinky]

        index  = f[1]
        middle = f[2]
        ring   = f[3]
        pinky  = f[4]

        total_up = sum(f[1:])  # ignore thumb for gesture logic

        if total_up == 0:
            return "fist"
        if index and not middle and not ring and not pinky:
            return "draw"
        if index and middle and not ring and not pinky:
            return "select"
        if index and middle and ring and not pinky:
            return "erase"

        return "unknown"
