import cv2
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, deque
from cv2 import dnn
from math import ceil
from stress_feature_extractor import StressFeatureExtractor

# ===============================
# Model constants and parameters
# ===============================
IMG_MEAN = np.array([127, 127, 127])
IMG_STD = 128.0
CENTER_VAR = 0.1
SIZE_VAR = 0.2
STRIDES = [8.0, 16.0, 32.0, 64.0]
THRESHOLD = 0.5
MIN_BOXES = [
    [10.0, 16.0, 24.0],
    [32.0, 48.0],
    [64.0, 96.0],
    [128.0, 192.0, 256.0]
]

# ===============================
# Utility Functions
# ===============================
def make_priors(image_shape):
    priors = []
    for stride, boxes in zip(STRIDES, MIN_BOXES):
        feat_h = int(ceil(image_shape[1] / stride))
        feat_w = int(ceil(image_shape[0] / stride))
        for y in range(feat_h):
            for x in range(feat_w):
                cx = (x + 0.5) / feat_w
                cy = (y + 0.5) / feat_h
                for size in boxes:
                    w = size / image_shape[0]
                    h = size / image_shape[1]
                    priors.append([cx, cy, w, h])
    print(f"Generated {len(priors)} priors.")
    return np.clip(np.array(priors), 0.0, 1.0)

def decode_locations(locations, priors, center_var, size_var):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_var * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_var) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)

def center_to_corners(boxes):
    return np.concatenate([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2
    ], len(boxes.shape) - 1)

def calc_iou(boxes1, boxes2, eps=1e-5):
    inter_tl = np.maximum(boxes1[..., :2], boxes2[..., :2])
    inter_br = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = np.clip(inter_br - inter_tl, 0, None)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    return inter_area / (area1 + area2 - inter_area + eps)

def nms_hard(boxes_scores, iou_thresh, top_k=-1):
    scores = boxes_scores[:, -1]
    boxes = boxes_scores[:, :-1]
    selected = []
    indices = np.argsort(scores)
    while len(indices) > 0:
        current = indices[-1]
        selected.append(current)
        if 0 < top_k == len(selected) or len(indices) == 1:
            break
        current_box = boxes[current, :]
        indices = indices[:-1]
        rest_boxes = boxes[indices, :]
        iou = calc_iou(rest_boxes, np.expand_dims(current_box, axis=0))
        indices = indices[iou <= iou_thresh]
    return boxes_scores[selected, :]

def filter_predictions(width, height, confidences, boxes, prob_thresh):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_boxes = []
    for i in range(1, confidences.shape[1]):
        probs = confidences[:, i]
        mask = probs > prob_thresh
        probs = probs[mask]
        if probs.size == 0:
            continue
        sub_boxes = boxes[mask, :]
        box_probs = np.concatenate([sub_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = nms_hard(box_probs, iou_thresh=0.3)
        picked_boxes.append(box_probs)
    if not picked_boxes:
        return np.array([])
    picked_boxes = np.concatenate(picked_boxes)
    picked_boxes[:, [0, 2]] *= width
    picked_boxes[:, [1, 3]] *= height
    return picked_boxes[:, :4].astype(np.int32)

# ===============================
# Valence–Arousal Estimation
# ===============================
def estimate_valence_arousal(features):
    ear = features['eye_aspect_ratio']
    mar = features['mouth_aspect_ratio']
    motion = features['head_motion']
    tension = features['forehead_tension']
    ear_norm = np.clip((ear - 0.15) / 0.15, 0, 1)
    mar_norm = np.clip((mar - 0.3) / 0.5, 0, 1)
    motion_norm = np.clip(motion / 0.2, 0, 1)
    tension_norm = np.clip(tension / 100, 0, 1)
    arousal = (1 - ear_norm) * 0.4 + mar_norm * 0.3 + motion_norm * 0.2 + tension_norm * 0.1
    valence = 0.6 * (1 - tension_norm) + 0.3 * ear_norm - 0.2 * mar_norm
    arousal, valence = np.clip(arousal, 0, 1), np.clip(valence, 0, 1)
    if valence > 0.6 and arousal < 0.4:
        mood, color = "Calm", (0, 255, 0)
    elif valence < 0.4 and arousal > 0.6:
        mood, color = "Stressed", (0, 0, 255)
    else:
        mood, color = "Focused", (0, 255, 255)
    return valence, arousal, mood, color

# ===============================
# Visualization
# ===============================
def generate_summary_plots():
    df = pd.read_csv('stress_features.csv')
    plt.figure(figsize=(8, 6))
    for mood, color in zip(["Calm", "Focused", "Stressed"], ["green", "yellow", "red"]):
        subset = df[df["Mood"] == mood]
        plt.scatter(subset["Valence"], subset["Arousal"], c=color, label=mood, alpha=0.6)
    plt.title("Valence–Arousal Emotional Mapping")
    plt.xlabel("Valence (Pleasantness → Positive)")
    plt.ylabel("Arousal (Activation → Energetic)")
    plt.legend(title="Mood Zone")
    plt.grid(True)
    plt.savefig("valence_arousal_plot.png")
    print("Saved valence_arousal_plot.png")

    plt.figure(figsize=(10, 4))
    df["Timestamp"] -= df["Timestamp"].iloc[0]
    plt.plot(df["Timestamp"], df["Valence"], label="Valence", color="green")
    plt.plot(df["Timestamp"], df["Arousal"], label="Arousal", color="orange")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Level")
    plt.title("Valence–Arousal Timeline")
    plt.legend()
    plt.grid(True)
    plt.savefig("valence_arousal_timeline.png")
    print("Saved valence_arousal_timeline.png")

# ===============================
# Main Function
# ===============================
def run_emotion_detection():
    emotions = {
        0: 'neutral', 1: 'happy', 2: 'surprised',
        3: 'sad', 4: 'angry', 5: 'disgusted', 6: 'fearful'
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access camera.")
        return

    w, h = int(cap.get(3)), int(cap.get(4))
    video_writer = cv2.VideoWriter('emotion_output.avi',
                                   cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))
    emotion_net = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')
    detector = dnn.readNetFromCaffe('RFB-320.prototxt', 'RFB-320.caffemodel')
    feature_extractor = StressFeatureExtractor()
    priors = make_priors([320, 240])

    with open('stress_features.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Emotion', 'EAR', 'MAR', 'Motion', 'Tension', 'Valence', 'Arousal', 'Mood'])

    emotion_times = defaultdict(float)
    current_emotion, emotion_start_time = None, time.time()
    start_time = time.time()

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 3))
    valence_data, arousal_data = deque(maxlen=100), deque(maxlen=100)
    val_line, = ax.plot([], [], label='Valence', color='green')
    aro_line, = ax.plot([], [], label='Arousal', color='orange')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.set_title("Live Valence–Arousal Tracking")
    ax.legend()

    print("Recording... Press 'q' to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        resized = cv2.resize(orig, (320, 240))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        detector.setInput(dnn.blobFromImage(rgb_frame, 1 / IMG_STD, (320, 240), 127))
        boxes, scores = detector.forward(["boxes", "scores"])
        boxes = np.expand_dims(boxes.reshape(-1, 4), axis=0)
        scores = np.expand_dims(scores.reshape(-1, 2), axis=0)
        boxes = decode_locations(boxes, priors, CENTER_VAR, SIZE_VAR)
        boxes = center_to_corners(boxes)
        boxes = filter_predictions(orig.shape[1], orig.shape[0], scores, boxes, THRESHOLD)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_emotion = None

        for (x1, y1, x2, y2) in boxes:
            face_region = gray_frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue

            face_roi = cv2.resize(face_region, (64, 64))
            face_blob = face_roi.reshape(1, 1, 64, 64)
            emotion_net.setInput(face_blob)
            preds = emotion_net.forward()
            detected_emotion = emotions[np.argmax(preds[0])]

            features = feature_extractor.extract_features(frame)
            if features:
                valence, arousal, mood, color = estimate_valence_arousal(features)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{detected_emotion}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Mood: {mood}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Valence: {valence:.2f}  Arousal: {arousal:.2f}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Track emotion duration
                if current_emotion is None:
                    current_emotion, emotion_start_time = detected_emotion, time.time()
                elif detected_emotion != current_emotion:
                    emotion_times[current_emotion] += time.time() - emotion_start_time
                    current_emotion, emotion_start_time = detected_emotion, time.time()

                # Update graph
                valence_data.append(valence)
                arousal_data.append(arousal)
                val_line.set_data(range(len(valence_data)), valence_data)
                aro_line.set_data(range(len(arousal_data)), arousal_data)
                plt.pause(0.001)

                # Log data
                with open('stress_features.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time(), detected_emotion,
                                     features['eye_aspect_ratio'], features['mouth_aspect_ratio'],
                                     features['head_motion'], features['forehead_tension'],
                                     round(valence, 3), round(arousal, 3), mood])

        video_writer.write(frame)
        cv2.imshow("Emotion & Valence–Arousal Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Final emotion duration update
    if current_emotion:
        emotion_times[current_emotion] += time.time() - emotion_start_time

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    plt.ioff()

    # Save emotion duration log
    with open('emotion_duration_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Emotion', 'Minutes', 'Seconds'])
        for emo, secs in emotion_times.items():
            writer.writerow([emo, int(secs // 60), int(secs % 60)])

    print("\nRecording finished.")
    generate_summary_plots()
    print("Results saved: emotion_output.avi, stress_features.csv, valence_arousal_plot.png, valence_arousal_timeline.png, emotion_duration_log.csv")

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    run_emotion_detection()
