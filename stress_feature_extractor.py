import cv2
import numpy as np
from scipy.spatial import distance as dist

class StressFeatureExtractor:
    def __init__(self):
        # Load face mesh / landmark detector (using mediapipe for precision)
        import mediapipe as mp
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.mp_draw = mp.solutions.drawing_utils

        # Previous frame landmarks (for head movement)
        self.prev_landmarks = None

    def get_eye_aspect_ratio(self, landmarks, frame_w, frame_h):
        # Eye indices (left eye)
        left_eye = [33, 160, 158, 133, 153, 144]
        pts = np.array([(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in left_eye])
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        C = dist.euclidean(pts[0], pts[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def get_mouth_aspect_ratio(self, landmarks, frame_w, frame_h):
        # Mouth indices
        mouth = [61, 81, 13, 14, 17, 0]
        pts = np.array([(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in mouth])
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        C = dist.euclidean(pts[0], pts[3])
        mar = (A + B) / (2.0 * C)
        return mar

    def get_head_motion(self, landmarks, frame_w, frame_h):
        if self.prev_landmarks is None:
            self.prev_landmarks = [(lm.x, lm.y) for lm in landmarks]
            return 0.0
        curr = np.array([(lm.x, lm.y) for lm in landmarks])
        prev = np.array(self.prev_landmarks)
        diff = np.linalg.norm(curr - prev)
        self.prev_landmarks = curr
        return diff

    def get_forehead_tension(self, frame, landmarks, frame_w, frame_h):
        # Approx forehead area using landmarks (region between eyebrows)
        region = [9, 107, 66, 105, 336, 296, 334, 10]
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array([(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in region])
        cv2.fillPoly(mask, [pts], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grad = cv2.Laplacian(gray, cv2.CV_64F).var()  # tension proxy
        return grad

    def extract_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        result = self.face_mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        ear = self.get_eye_aspect_ratio(landmarks, w, h)
        mar = self.get_mouth_aspect_ratio(landmarks, w, h)
        motion = self.get_head_motion(landmarks, w, h)
        tension = self.get_forehead_tension(frame, landmarks, w, h)

        return {
            "eye_aspect_ratio": ear,
            "mouth_aspect_ratio": mar,
            "head_motion": motion,
            "forehead_tension": tension
        }
