import cv2
import numpy as np

class Player:
    def __init__(self, id: int, face_filter: np.ndarray | None = None):
        self.id = id
        self.face_filter = face_filter

    def is_moving(self, current_face: np.ndarray) -> bool:
        if self.face_filter is None or current_face is None:
            return False

        # Resize current face to match stored size just in case
        current_face = cv2.resize(current_face, (100, 100))

        # Calculate absolute difference
        diff = cv2.absdiff(self.face_filter, current_face)

        # Compute mean pixel difference
        score = np.mean(diff)

        print(f"[Player {self.id}] Difference Score: {score}")

        # Threshold — tweak this based on testing
        return score > 25  # lower = more sensitive

    def is_won(self) -> bool:
        # TODO: interface with CV scriptx
        # this is a stub
        # how to do this???
        return False