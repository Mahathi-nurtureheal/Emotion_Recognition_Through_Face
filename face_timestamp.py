import cv2
import numpy as np
import time
import csv
from collections import defaultdict
from cv2 import dnn
from math import ceil

# Model constants and hyperparameters
IMG_MEAN = np.array([127, 127, 127])
IMG_STD = 128.0
IOU_THRESH = 0.3
CENTER_VAR = 0.1
SIZE_VAR = 0.2
STRIDES = [8.0, 16.0, 32.0, 64.0]
THRESHOLD = 0.5

# Default anchor box sizes
MIN_BOXES = [
    [10.0, 16.0, 24.0],
    [32.0, 48.0],
    [64.0, 96.0],
    [128.0, 192.0, 256.0]
]


def make_priors(image_shape):
    feature_maps = []
    shrink_list = []

    for size in image_shape:
        feature_maps.append([int(ceil(size / stride)) for stride in STRIDES])
        shrink_list.append(STRIDES)

    priors = []
    for idx in range(len(feature_maps[0])):
        scale_w = image_shape[0] / shrink_list[0][idx]
        scale_h = image_shape[1] / shrink_list[1][idx]

        for y in range(feature_maps[1][idx]):
            for x in range(feature_maps[0][idx]):
                cx = (x + 0.5) / scale_w
                cy = (y + 0.5) / scale_h
                for box in MIN_BOXES[idx]:
                    w = box / image_shape[0]
                    h = box / image_shape[1]
                    priors.append([cx, cy, w, h])

    print(f"Generated {len(priors)} priors.")
    return np.clip(priors, 0.0, 1.0)


def calc_area(tl, br):
    wh = np.clip(br - tl, 0.0, None)
    return wh[..., 0] * wh[..., 1]


def calc_iou(boxes1, boxes2, eps=1e-5):
    inter_tl = np.maximum(boxes1[..., :2], boxes2[..., :2])
    inter_br = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_area = calc_area(inter_tl, inter_br)
    area1 = calc_area(boxes1[..., :2], boxes1[..., 2:])
    area2 = calc_area(boxes2[..., :2], boxes2[..., 2:])
    return inter_area / (area1 + area2 - inter_area + eps)


def nms_hard(boxes_scores, iou_thresh, top_k=-1, candidate_size=200):
    scores = boxes_scores[:, -1]
    boxes = boxes_scores[:, :-1]
    selected = []
    indices = np.argsort(scores)[-candidate_size:]

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


def filter_predictions(width, height, confidences, boxes, prob_thresh, iou_thresh=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_boxes, picked_labels = [], []

    for i in range(1, confidences.shape[1]):
        probs = confidences[:, i]
        mask = probs > prob_thresh
        probs = probs[mask]
        if probs.size == 0:
            continue
        sub_boxes = boxes[mask, :]
        box_probs = np.concatenate([sub_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = nms_hard(box_probs, iou_thresh=iou_thresh, top_k=top_k)
        picked_boxes.append(box_probs)
        picked_labels.extend([i] * box_probs.shape[0])

    if not picked_boxes:
        return np.array([]), np.array([]), np.array([])

    picked_boxes = np.concatenate(picked_boxes)
    picked_boxes[:, [0, 2]] *= width
    picked_boxes[:, [1, 3]] *= height
    return picked_boxes[:, :4].astype(np.int32), np.array(picked_labels), picked_boxes[:, 4]


# MAIN FUNCTION

def run_emotion_detection():
    emotions = {
        0: 'neutral', 1: 'happy', 2: 'surprised',
        3: 'sad', 4: 'angry', 5: 'disgusted', 6: 'fearful'
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access camera or video.")
        return

    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter('emotion_output.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))

    emotion_net = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')
    detector = dnn.readNetFromCaffe('RFB-320.prototxt', 'RFB-320.caffemodel')

    input_dim = [320, 240]
    priors = make_priors(input_dim)

    emotion_times = defaultdict(float)
    current_emotion = None
    emotion_start_time = time.time()

    print("Recording... Press 'q' to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        resized = cv2.resize(orig, tuple(input_dim))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        detector.setInput(dnn.blobFromImage(rgb_frame, 1 / IMG_STD, tuple(input_dim), 127))
        boxes, scores = detector.forward(["boxes", "scores"])
        boxes = np.expand_dims(boxes.reshape(-1, 4), axis=0)
        scores = np.expand_dims(scores.reshape(-1, 2), axis=0)

        boxes = decode_locations(boxes, priors, CENTER_VAR, SIZE_VAR)
        boxes = center_to_corners(boxes)
        boxes, labels, probs = filter_predictions(
            orig.shape[1], orig.shape[0], scores, boxes, THRESHOLD)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_emotion = None
        for (x1, y1, x2, y2) in boxes:
            w_box, h_box = x2 - x1, y2 - y1
            face_roi = cv2.resize(gray_frame[y1:y1 + h_box, x1:x1 + w_box], (64, 64))
            face_blob = face_roi.reshape(1, 1, 64, 64)
            emotion_net.setInput(face_blob)
            preds = emotion_net.forward()
            detected_emotion = emotions[np.argmax(preds[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (215, 5, 247), 2)
            cv2.putText(frame, f"{detected_emotion}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (215, 5, 247), 2)

        # Emotion duration tracking
        if detected_emotion:
            if current_emotion is None:
                current_emotion = detected_emotion
                emotion_start_time = time.time()
            elif detected_emotion != current_emotion:
                duration = time.time() - emotion_start_time
                emotion_times[current_emotion] += duration
                current_emotion = detected_emotion
                emotion_start_time = time.time()

        writer.write(frame)
        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Record last emotion time
    if current_emotion:
        emotion_times[current_emotion] += time.time() - emotion_start_time

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Save to CSV file
    with open('emotion_duration_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Emotion', 'Minutes', 'Seconds'])
        for emo, sec in emotion_times.items():
            mins = int(sec // 60)
            secs = int(sec % 60)
            writer.writerow([emo, mins, secs])

    print("\n=== Emotion Duration Summary ===")
    for emo, sec in emotion_times.items():
        mins = int(sec // 60)
        secs = int(sec % 60)
        print(f"{emo}: {mins} min {secs} sec")
    print("===============================")
    print("\nResults saved in 'emotion_duration_log.csv'")

if __name__ == "__main__":
    run_emotion_detection()
