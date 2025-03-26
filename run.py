import cv2
import numpy as np
import csv
import time
import math
import os
from ultralytics import YOLO  # Make sure to install ultralytics (pip install ultralytics)

# ----------------- Model & Vehicle Setup -----------------
# Load your YOLOv11 model (in .pt format)
model = YOLO("yolo/yolo11n.pt")

# The model’s names dictionary contains class names (usually COCO classes)
# Define the vehicle-related classes you want to count
vehicle_classes = {"car", "bus", "truck", "motorcycle"}

# ----------------- CSV Logging -----------------
csv_file = open("vehicle_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Vehicle ID", "Crossing Time", "Line", "Area", "Speed (pixels/sec)", "Object Type"])

# ----------------- Helper Functions -----------------
def get_centroid(x, y, w, h):
    """Return the centroid of a bounding box."""
    return (x + w // 2, y + h // 2)

def point_line_distance(point, line_start, line_end):
    """Compute the perpendicular distance from a point to a line."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return num / den if den != 0 else 0

def projection_ratio(point, line_start, line_end):
    """
    Compute the projection ratio of a point onto a line segment.
    Returns a value between 0 and 1 indicating where the perpendicular
    projection of the point falls relative to the segment.
    """
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    line_vec = b - a
    if np.linalg.norm(line_vec) == 0:
        return 0
    return np.dot(p - a, line_vec) / np.dot(line_vec, line_vec)

# ----------------- Tracking Setup -----------------
tracked_vehicles = {}   # key: vehicle id, value: track info
next_vehicle_id = 1

def update_track(centroid, current_time, match_threshold=50, time_threshold=0.5):
    """
    Update tracked vehicles with a new detection (centroid).
    Returns the vehicle id for a matched or new detection.
    """
    global next_vehicle_id, tracked_vehicles
    matched_id = None
    for vid, track in tracked_vehicles.items():
        if current_time - track["last_time"] > time_threshold:
            continue
        distance = math.hypot(centroid[0] - track["last_pos"][0],
                              centroid[1] - track["last_pos"][1])
        if distance < match_threshold:
            matched_id = vid
            track["last_time"] = current_time
            # Smooth the centroid position
            track["last_pos"] = ((track["last_pos"][0] + centroid[0]) // 2,
                                 (track["last_pos"][1] + centroid[1]) // 2)
            break
    if matched_id is None:
        tracked_vehicles[next_vehicle_id] = {
            "start_time": current_time,
            "start_pos": centroid,
            "last_time": current_time,
            "last_pos": centroid,
            "crossed": False
        }
        matched_id = next_vehicle_id
        next_vehicle_id += 1
    return matched_id

def compute_speed(track):
    """Compute speed (pixels/sec) from track data."""
    dt = track["last_time"] - track["start_time"]
    if dt <= 0:
        return 0
    dx = track["last_pos"][0] - track["start_pos"][0]
    dy = track["last_pos"][1] - track["start_pos"][1]
    return math.hypot(dx, dy) / dt

# ----------------- Diagonal Line Settings -----------------
# Define two detection diagonal lines
diag1_detection_start = (100, 50)
diag1_detection_end   = (2400, 1100)
diag2_detection_start = (1100, 50)
diag2_detection_end   = (900, 1300)

# Parameters for determining if a centroid is "near" a line
offset = 10
padding_ratio_min = 0.1
padding_ratio_max = 0.8

# ----------------- Video Setup -----------------
source_video = '/Users/marsman/custom/github/vehicle_counting_tensorflow/traffic4way2.mp4'
cap = cv2.VideoCapture(source_video)
total_vehicles = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error encountered.")
        break

    frame_count += 1
    # Process every 4th frame to speed up detection
    if frame_count % 4 != 0:
        continue

    height, width, channels = frame.shape

    # Run YOLO inference on the frame using the Ultralytics API
    results = model(frame)  # Run inference; results is a list (one per image)
    # Initialize lists to hold detection data
    boxes = []
    confidences = []
    class_ids = []

    # Iterate through detected boxes
    for box in results[0].boxes:
        # Get bounding box coordinates (x1, y1, x2, y2)
        coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, coords)
        w = x2 - x1
        h = y2 - y1
        cls_id = int(box.cls.cpu().numpy())
        label = results[0].names[cls_id]
        confidence = float(box.conf.cpu().numpy())
        if confidence > 0.5 and label in vehicle_classes:
            boxes.append([x1, y1, w, h])
            confidences.append(confidence)
            class_ids.append(cls_id)

    # Get the current time in seconds from the video
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Process each detection
    for i, b in enumerate(boxes):
        x, y, w, h = b
        centroid = get_centroid(x, y, w, h)

        # Compute distances and projection ratios for both diagonal lines
        dist1 = point_line_distance(centroid, diag1_detection_start, diag1_detection_end)
        dist2 = point_line_distance(centroid, diag2_detection_start, diag2_detection_end)
        proj_ratio1 = projection_ratio(centroid, diag1_detection_start, diag1_detection_end)
        proj_ratio2 = projection_ratio(centroid, diag2_detection_start, diag2_detection_end)
        near_diag1 = (dist1 < offset) and (padding_ratio_min <= proj_ratio1 <= padding_ratio_max)
        near_diag2 = (dist2 < offset) and (padding_ratio_min <= proj_ratio2 <= padding_ratio_max)

        # Update or create track for this detected vehicle
        vid = update_track(centroid, current_time)
        track = tracked_vehicles[vid]

        # If the detection is near one of the lines and hasn't been counted yet:
        if (near_diag1 or near_diag2) and (not track["crossed"]):
            track["crossed"] = True
            speed = compute_speed(track)
            total_vehicles += 1
            line_used = "line 1" if near_diag1 else "line 2"
            area = w * h
            csv_writer.writerow([vid,
                                 time.strftime("%M:%S", time.gmtime(current_time)),
                                 line_used,
                                 area,
                                 f"{speed:.2f}",
                                 results[0].names[class_ids[i]]])
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        # Draw the detection bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{results[0].names[class_ids[i]]} {confidences[i]:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the diagonal detection lines for reference
    cv2.line(frame, diag1_detection_start, diag1_detection_end, (255, 0, 0), 2)
    cv2.line(frame, diag2_detection_start, diag2_detection_end, (255, 0, 0), 2)
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
