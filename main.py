################################## VERSION 4 ##################################

import cv2
import numpy as np
import csv
import time
import math

# Parameters
min_contour_width = 40  
min_contour_height = 40  
offset = 10                # Maximum distance from the line to consider a crossing
padding_ratio_min = 0.1    # For drawing: start of visible line (10% from start)
padding_ratio_max = 0.8    # For drawing: end of visible line (80% from start)

# Global vehicle counter for logging and a tracker for active vehicles
vehicles = 0
tracked_vehicles = {}    # key: vehicle id, value: dict with start_time, start_pos, last_time, last_pos, crossed
next_vehicle_id = 1      # counter to assign new vehicle IDs

# Open CSV file for logging
csv_file = open("vehicle_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Vehicle ID", "Crossing Time", "Line", "Area", "Speed (pixels/sec)"])

def get_centroid(x, y, w, h):
    """Compute the centroid of a rectangle."""
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return (cx, cy)

def point_line_distance(point, line_start, line_end):
    """Compute the perpendicular distance from a point to a line."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / den

def projection_ratio(point, line_start, line_end):
    """
    Compute the projection ratio of a point onto a line segment.
    Returns a scalar between 0 and 1 representing where the perpendicular
    projection of the point falls relative to the line segment.
    """
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_vec = line_end - line_start
    if np.linalg.norm(line_vec) == 0:
        return 0
    t = np.dot(point - line_start, line_vec) / np.dot(line_vec, line_vec)
    return t

def filter_contained_boxes(boxes, threshold=0.9):
    """
    Filters out boxes that are mostly (>= threshold) contained in another box.
    Each box is represented as [x, y, w, h].
    """
    filtered = []
    for i, box_i in enumerate(boxes):
        x1, y1, w1, h1 = box_i
        area_i = w1 * h1
        keep = True
        for j, box_j in enumerate(boxes):
            if i == j:
                continue
            x2, y2, w2, h2 = box_j
            # Calculate the intersection area between box_i and box_j
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h
            if area_i > 0 and (intersection / area_i) >= threshold:
                keep = False
                break
        if keep:
            filtered.append(box_i)
    return filtered

def get_draw_line(full_start, full_end, ratio_min, ratio_max):
    """Calculate endpoints for the drawn (padded) line."""
    full_start = np.array(full_start)
    full_end = np.array(full_end)
    draw_start = full_start + (full_end - full_start) * ratio_min
    draw_end   = full_start + (full_end - full_start) * ratio_max
    return tuple(draw_start.astype(int)), tuple(draw_end.astype(int))

def update_track(detection, current_time, match_threshold=30):
    """
    Update the tracked_vehicles with a new detection (centroid).
    Returns the vehicle id for the matched or new detection.
    """
    global next_vehicle_id, tracked_vehicles
    matched_id = None
    for vid, track in tracked_vehicles.items():
        # Compute distance from detection to last known position
        if math.hypot(detection[0]-track["last_pos"][0], detection[1]-track["last_pos"][1]) < match_threshold:
            matched_id = vid
            # Update track with latest detection
            track["last_time"] = current_time
            track["last_pos"] = detection
            break
    if matched_id is None:
        # Create a new track
        tracked_vehicles[next_vehicle_id] = {
            "start_time": current_time,
            "start_pos": detection,
            "last_time": current_time,
            "last_pos": detection,
            "crossed": False
        }
        matched_id = next_vehicle_id
        next_vehicle_id += 1
    return matched_id

def compute_speed(track):
    """Compute speed in pixels/sec from the track data."""
    dt = track["last_time"] - track["start_time"]
    if dt <= 0:
        return 0
    dx = track["last_pos"][0] - track["start_pos"][0]
    dy = track["last_pos"][1] - track["start_pos"][1]
    distance = math.hypot(dx, dy)
    return distance / dt

# -------------------- Manual Line Settings --------------------
diag1_detection_start = (100, 50)      # First detection line: start point
diag1_detection_end   = (2400, 1100)     # First detection line: end point

diag2_detection_start = (1100, 50)       # Second detection line: start point
diag2_detection_end   = (900, 1300)       # Second detection line: end point

diag1_draw_start, diag1_draw_end = get_draw_line(diag1_detection_start, diag1_detection_end,
                                                   padding_ratio_min, padding_ratio_max)
diag2_draw_start, diag2_draw_end = get_draw_line(diag2_detection_start, diag2_detection_end,
                                                   padding_ratio_min, padding_ratio_max)
# ---------------------------------------------------------------

# Open video
cap = cv2.VideoCapture('traffic4way.mp4')
cap.set(3, 1920)
cap.set(4, 1080)

if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while ret:
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret2, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Collect valid bounding boxes for the current frame
    boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w < min_contour_width or h < min_contour_height:
            continue
        boxes.append([x, y, w, h])
    
    boxes = filter_contained_boxes(boxes, threshold=0.9)
    
    # Draw the drawn lines (padded)
    cv2.line(frame1, diag1_draw_start, diag1_draw_end, (0, 255, 0), 2)
    cv2.line(frame1, diag2_draw_start, diag2_draw_end, (0, 255, 0), 2)
    
    # Get current video time in seconds
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    time_str = time.strftime("%M:%S", time.gmtime(current_time))
    
    for (x, y, w, h) in boxes:
        # Draw bounding box for visualization
        cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
        
        # Process splitting if box is wide (may contain two vehicles)
        if w > 1.5 * min_contour_width:
            est_count = int(round(w / (1.5 * min_contour_width)))
            sub_box_width = w / est_count
            for i in range(est_count):
                new_cx = x + int((i + 0.5) * sub_box_width)
                new_cy = y + int(h / 2)
                new_centroid = (new_cx, new_cy)
                sub_area = int(sub_box_width * h)
                
                # Update tracker for this sub-detection
                vid = update_track(new_centroid, current_time)
                track = tracked_vehicles[vid]
                
                # Check crossing criteria using full detection lines
                dist1 = point_line_distance(new_centroid, diag1_detection_start, diag1_detection_end)
                dist2 = point_line_distance(new_centroid, diag2_detection_start, diag2_detection_end)
                proj_ratio1 = projection_ratio(new_centroid, diag1_detection_start, diag1_detection_end)
                proj_ratio2 = projection_ratio(new_centroid, diag2_detection_start, diag2_detection_end)
                near_diag1 = (dist1 < offset) and (padding_ratio_min <= proj_ratio1 <= padding_ratio_max)
                near_diag2 = (dist2 < offset) and (padding_ratio_min <= proj_ratio2 <= padding_ratio_max)
                
                if (near_diag1 or near_diag2) and (not track["crossed"]):
                    # Mark track as crossed and compute speed
                    track["crossed"] = True
                    speed = compute_speed(track)
                    vehicles += 1
                    line_used = "line_1" if near_diag1 else "line_2"
                    csv_writer.writerow([vehicles, time_str, line_used, sub_area, f"{speed:.2f}"])
                    cv2.circle(frame1, new_centroid, 5, (0, 0, 255), -1)
        else:
            centroid = get_centroid(x, y, w, h)
            # Update tracker for this detection
            vid = update_track(centroid, current_time)
            track = tracked_vehicles[vid]
            
            dist1 = point_line_distance(centroid, diag1_detection_start, diag1_detection_end)
            dist2 = point_line_distance(centroid, diag2_detection_start, diag2_detection_end)
            proj_ratio1 = projection_ratio(centroid, diag1_detection_start, diag1_detection_end)
            proj_ratio2 = projection_ratio(centroid, diag2_detection_start, diag2_detection_end)
            near_diag1 = (dist1 < offset) and (padding_ratio_min <= proj_ratio1 <= padding_ratio_max)
            near_diag2 = (dist2 < offset) and (padding_ratio_min <= proj_ratio2 <= padding_ratio_max)
            
            if (near_diag1 or near_diag2) and (not track["crossed"]):
                track["crossed"] = True
                speed = compute_speed(track)
                vehicles += 1
                line_used = "line_1" if near_diag1 else "line_2"
                area = w * h
                csv_writer.writerow([vehicles, time_str, line_used, area, f"{speed:.2f}"])
                cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
    
    cv2.putText(frame1, f"Total Vehicles: {vehicles}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
    cv2.imshow("Vehicle Detection", frame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame1 = frame2
    ret, frame2 = cap.read()

# Clean up
csv_file.close()
cv2.destroyAllWindows()
cap.release()




################################## VERSION 3 ##################################


# import cv2
# import numpy as np

# # Parameters
# min_contour_width = 40  
# min_contour_height = 40  
# offset = 10                # Maximum distance from the line to consider a crossing
# padding_ratio_min = 0.1    # For drawing: start of visible line (10% from start)
# padding_ratio_max = 0.8    # For drawing: end of visible line (80% from start)
# vehicles = 0

# # List to store centroids already counted
# already_counted = []

# def get_centroid(x, y, w, h):
#     """Compute the centroid of a rectangle."""
#     cx = x + int(w / 2)
#     cy = y + int(h / 2)
#     return (cx, cy)

# def point_line_distance(point, line_start, line_end):
#     """Compute the perpendicular distance from a point to a line."""
#     x0, y0 = point
#     x1, y1 = line_start
#     x2, y2 = line_end
#     num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
#     den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
#     return num / den

# def projection_ratio(point, line_start, line_end):
#     """
#     Compute the projection ratio of a point onto a line segment.
#     Returns a scalar between 0 and 1 representing where the perpendicular
#     projection of the point falls relative to the line segment.
#     """
#     point = np.array(point)
#     line_start = np.array(line_start)
#     line_end = np.array(line_end)
#     line_vec = line_end - line_start
#     if np.linalg.norm(line_vec) == 0:
#         return 0
#     t = np.dot(point - line_start, line_vec) / np.dot(line_vec, line_vec)
#     return t

# def is_new_detection(centroid, detected_list, threshold):
#     """Check if this centroid is far enough from any previously counted centroid."""
#     for pt in detected_list:
#         if np.linalg.norm(np.array(centroid) - np.array(pt)) < threshold:
#             return False
#     return True

# def filter_contained_boxes(boxes, threshold=0.9):
#     """
#     Filters out boxes that are mostly (>= threshold) contained in another box.
#     Each box is represented as [x, y, w, h].
#     """
#     filtered = []
#     for i, box_i in enumerate(boxes):
#         x1, y1, w1, h1 = box_i
#         area_i = w1 * h1
#         keep = True
#         for j, box_j in enumerate(boxes):
#             if i == j:
#                 continue
#             x2, y2, w2, h2 = box_j
#             # Calculate the intersection area between box_i and box_j
#             inter_x1 = max(x1, x2)
#             inter_y1 = max(y1, y2)
#             inter_x2 = min(x1 + w1, x2 + w2)
#             inter_y2 = min(y1 + h1, y2 + h2)
#             inter_w = max(0, inter_x2 - inter_x1)
#             inter_h = max(0, inter_y2 - inter_y1)
#             intersection = inter_w * inter_h
#             if area_i > 0 and (intersection / area_i) >= threshold:
#                 # If box_i is mostly contained in box_j, ignore it.
#                 keep = False
#                 break
#         if keep:
#             filtered.append(box_i)
#     return filtered

# # -------------------- Manual Line Settings --------------------
# # Tune these endpoints as needed for your video.
# diag1_detection_start = (100, 50)      # First detection line: start point
# diag1_detection_end   = (2400, 1100)     # First detection line: end point

# diag2_detection_start = (1100, 50)       # Second detection line: start point
# diag2_detection_end   = (900, 1300)       # Second detection line: end point

# # For drawing, we only show the central portion of the detection lines.
# def get_draw_line(full_start, full_end, ratio_min, ratio_max):
#     full_start = np.array(full_start)
#     full_end = np.array(full_end)
#     draw_start = full_start + (full_end - full_start) * ratio_min
#     draw_end   = full_start + (full_end - full_start) * ratio_max
#     return tuple(draw_start.astype(int)), tuple(draw_end.astype(int))

# diag1_draw_start, diag1_draw_end = get_draw_line(diag1_detection_start, diag1_detection_end,
#                                                    padding_ratio_min, padding_ratio_max)
# diag2_draw_start, diag2_draw_end = get_draw_line(diag2_detection_start, diag2_detection_end,
#                                                    padding_ratio_min, padding_ratio_max)
# # ---------------------------------------------------------------

# # Open video
# cap = cv2.VideoCapture('/Users/marsman/custom/github/vehicle_counting_tensorflow/traffic4way2.mp4')
# cap.set(3, 1920)
# cap.set(4, 1080)

# if cap.isOpened():
#     ret, frame1 = cap.read()
# else:
#     ret = False

# ret, frame1 = cap.read()
# ret, frame2 = cap.read()

# while ret:
#     d = cv2.absdiff(frame1, frame2)
#     grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grey, (5, 5), 0)
#     ret2, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
#     dilated = cv2.dilate(th, np.ones((3, 3)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#     closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Collect valid bounding boxes for the current frame
#     boxes = []
#     for c in contours:
#         (x, y, w, h) = cv2.boundingRect(c)
#         if w < min_contour_width or h < min_contour_height:
#             continue
#         boxes.append([x, y, w, h])
    
#     # Filter out small boxes that are mostly contained in a larger one
#     boxes = filter_contained_boxes(boxes, threshold=0.9)
    
#     # Draw the padded (draw) lines on the frame
#     cv2.line(frame1, diag1_draw_start, diag1_draw_end, (0, 255, 0), 2)
#     cv2.line(frame1, diag2_draw_start, diag2_draw_end, (0, 255, 0), 2)
    
#     for (x, y, w, h) in boxes:
#         # Draw the bounding box for visualization
#         cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
        
#         # If the box width is significantly larger than min_contour_width,
#         # assume it may contain more than one vehicle and split it horizontally.
#         if w > 1.5 * min_contour_width:
#             est_count = int(round(w / (1.5 * min_contour_width)))
#             # Split the box horizontally into est_count regions.
#             for i in range(est_count):
#                 new_cx = x + int((i + 0.5) * (w / est_count))
#                 new_cy = y + int(h / 2)
#                 new_centroid = (new_cx, new_cy)
                
#                 # Check distances to the full detection lines
#                 dist1 = point_line_distance(new_centroid, diag1_detection_start, diag1_detection_end)
#                 dist2 = point_line_distance(new_centroid, diag2_detection_start, diag2_detection_end)
#                 # Calculate projection ratios on each full detection line
#                 proj_ratio1 = projection_ratio(new_centroid, diag1_detection_start, diag1_detection_end)
#                 proj_ratio2 = projection_ratio(new_centroid, diag2_detection_start, diag2_detection_end)
                
#                 near_diag1 = (dist1 < offset) and (padding_ratio_min <= proj_ratio1 <= padding_ratio_max)
#                 near_diag2 = (dist2 < offset) and (padding_ratio_min <= proj_ratio2 <= padding_ratio_max)
                
#                 if (near_diag1 or near_diag2) and is_new_detection(new_centroid, already_counted, threshold=30):
#                     vehicles += 1
#                     already_counted.append(new_centroid)
#                     cv2.circle(frame1, new_centroid, 5, (0, 0, 255), -1)
#         else:
#             # Use the normal centroid for a single vehicle detection
#             centroid = get_centroid(x, y, w, h)
#             cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
            
#             dist1 = point_line_distance(centroid, diag1_detection_start, diag1_detection_end)
#             dist2 = point_line_distance(centroid, diag2_detection_start, diag2_detection_end)
#             proj_ratio1 = projection_ratio(centroid, diag1_detection_start, diag1_detection_end)
#             proj_ratio2 = projection_ratio(centroid, diag2_detection_start, diag2_detection_end)
            
#             near_diag1 = (dist1 < offset) and (padding_ratio_min <= proj_ratio1 <= padding_ratio_max)
#             near_diag2 = (dist2 < offset) and (padding_ratio_min <= proj_ratio2 <= padding_ratio_max)
            
#             if (near_diag1 or near_diag2) and is_new_detection(centroid, already_counted, threshold=30):
#                 vehicles += 1
#                 already_counted.append(centroid)
    
#     cv2.putText(frame1, f"Total Vehicles: {vehicles}", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
#     cv2.imshow("Vehicle Detection", frame1)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
#     frame1 = frame2
#     ret, frame2 = cap.read()

# cv2.destroyAllWindows()
# cap.release()



################################## VERSION 2 ##################################


# import cv2
# import numpy as np

# # Parameters
# min_contour_width = 40  
# min_contour_height = 40  
# offset = 10                # Maximum distance from the line to consider a crossing
# padding_ratio_min = 0.1    # For drawing: start of visible line (10% from start)
# padding_ratio_max = 0.8    # For drawing: end of visible line (80% from start)
# vehicles = 0

# # List to store centroids already counted
# already_counted = []

# def get_centroid(x, y, w, h):
#     """Compute the centroid of a rectangle."""
#     cx = x + int(w / 2)
#     cy = y + int(h / 2)
#     return (cx, cy)

# def point_line_distance(point, line_start, line_end):
#     """Compute the perpendicular distance from a point to a line."""
#     x0, y0 = point
#     x1, y1 = line_start
#     x2, y2 = line_end
#     num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
#     den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
#     return num / den

# def projection_ratio(point, line_start, line_end):
#     """
#     Compute the projection ratio of a point onto a line segment.
#     Returns a scalar between 0 and 1 representing where the perpendicular
#     projection of the point falls relative to the line segment.
#     """
#     point = np.array(point)
#     line_start = np.array(line_start)
#     line_end = np.array(line_end)
#     line_vec = line_end - line_start
#     if np.linalg.norm(line_vec) == 0:
#         return 0
#     t = np.dot(point - line_start, line_vec) / np.dot(line_vec, line_vec)
#     return t

# def is_new_detection(centroid, detected_list, threshold):
#     """Check if this centroid is far enough from any previously counted centroid."""
#     for pt in detected_list:
#         if np.linalg.norm(np.array(centroid) - np.array(pt)) < threshold:
#             return False
#     return True

# def filter_contained_boxes(boxes, threshold=0.9):
#     """
#     Filters out boxes that are mostly (>= threshold) contained in another box.
#     Each box is represented as [x, y, w, h].
#     """
#     filtered = []
#     for i, box_i in enumerate(boxes):
#         x1, y1, w1, h1 = box_i
#         area_i = w1 * h1
#         keep = True
#         for j, box_j in enumerate(boxes):
#             if i == j:
#                 continue
#             x2, y2, w2, h2 = box_j
#             # Calculate the intersection area between box_i and box_j
#             inter_x1 = max(x1, x2)
#             inter_y1 = max(y1, y2)
#             inter_x2 = min(x1 + w1, x2 + w2)
#             inter_y2 = min(y1 + h1, y2 + h2)
#             inter_w = max(0, inter_x2 - inter_x1)
#             inter_h = max(0, inter_y2 - inter_y1)
#             intersection = inter_w * inter_h
#             if area_i > 0 and (intersection / area_i) >= threshold:
#                 # If box_i is mostly contained in box_j, ignore it.
#                 keep = False
#                 break
#         if keep:
#             filtered.append(box_i)
#     return filtered

# # -------------------- Manual Line Settings --------------------
# # Tune these endpoints as needed for your video.
# diag1_detection_start = (100, 50)      # First detection line: start point
# diag1_detection_end   = (2400, 1100)     # First detection line: end point

# diag2_detection_start = (1100, 50)       # Second detection line: start point
# diag2_detection_end   = (900, 1300)       # Second detection line: end point

# # For drawing, we only show the central portion of the detection lines.
# def get_draw_line(full_start, full_end, ratio_min, ratio_max):
#     full_start = np.array(full_start)
#     full_end = np.array(full_end)
#     draw_start = full_start + (full_end - full_start) * ratio_min
#     draw_end   = full_start + (full_end - full_start) * ratio_max
#     return tuple(draw_start.astype(int)), tuple(draw_end.astype(int))

# diag1_draw_start, diag1_draw_end = get_draw_line(diag1_detection_start, diag1_detection_end,
#                                                    padding_ratio_min, padding_ratio_max)
# diag2_draw_start, diag2_draw_end = get_draw_line(diag2_detection_start, diag2_detection_end,
#                                                    padding_ratio_min, padding_ratio_max)
# # ---------------------------------------------------------------

# # Open video
# cap = cv2.VideoCapture('/Users/marsman/custom/github/vehicle_counting_tensorflow/traffic4way2.mp4')
# cap.set(3, 1920)
# cap.set(4, 1080)

# if cap.isOpened():
#     ret, frame1 = cap.read()
# else:
#     ret = False

# ret, frame1 = cap.read()
# ret, frame2 = cap.read()

# while ret:
#     d = cv2.absdiff(frame1, frame2)
#     grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grey, (5, 5), 0)
#     ret2, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
#     dilated = cv2.dilate(th, np.ones((3, 3)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#     closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Collect valid bounding boxes for the current frame
#     boxes = []
#     for c in contours:
#         (x, y, w, h) = cv2.boundingRect(c)
#         if w < min_contour_width or h < min_contour_height:
#             continue
#         boxes.append([x, y, w, h])
    
#     # Filter out small boxes that are mostly contained in a larger one
#     boxes = filter_contained_boxes(boxes, threshold=0.9)
    
#     # Draw the padded (draw) lines on the frame
#     cv2.line(frame1, diag1_draw_start, diag1_draw_end, (0, 255, 0), 2)
#     cv2.line(frame1, diag2_draw_start, diag2_draw_end, (0, 255, 0), 2)
    
#     for (x, y, w, h) in boxes:
#         # Draw bounding box for visualization
#         cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
#         centroid = get_centroid(x, y, w, h)
#         cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
        
#         # Check distances to the full detection lines
#         dist1 = point_line_distance(centroid, diag1_detection_start, diag1_detection_end)
#         dist2 = point_line_distance(centroid, diag2_detection_start, diag2_detection_end)
        
#         # Calculate projection ratios on each full detection line
#         proj_ratio1 = projection_ratio(centroid, diag1_detection_start, diag1_detection_end)
#         proj_ratio2 = projection_ratio(centroid, diag2_detection_start, diag2_detection_end)
        
#         # Check if centroid is near a line and its projection falls within the padded region
#         near_diag1 = (dist1 < offset) and (padding_ratio_min <= proj_ratio1 <= padding_ratio_max)
#         near_diag2 = (dist2 < offset) and (padding_ratio_min <= proj_ratio2 <= padding_ratio_max)
        
#         if (near_diag1 or near_diag2) and is_new_detection(centroid, already_counted, threshold=30):
#             vehicles += 1
#             already_counted.append(centroid)
    
#     cv2.putText(frame1, f"Total Vehicles: {vehicles}", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
#     cv2.imshow("Vehicle Detection", frame1)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
#     frame1 = frame2
#     ret, frame2 = cap.read()

# cv2.destroyAllWindows()
# cap.release()



################################## VERSION 1 ##################################

# import cv2
# import numpy as np

# # Parameters
# min_contour_width = 40  
# min_contour_height = 40  
# offset = 10                # Maximum distance from the line to consider a crossing
# padding_ratio_min = 0.2    # Ignore vehicles if projection ratio is less than 20%
# padding_ratio_max = 0.8    # Ignore vehicles if projection ratio is more than 80%
# vehicles = 0

# # List to store centroids already counted
# already_counted = []

# def get_centroid(x, y, w, h):
#     """Compute the centroid of a rectangle."""
#     cx = x + int(w / 2)
#     cy = y + int(h / 2)
#     return (cx, cy)

# def point_line_distance(point, line_start, line_end):
#     """Compute the perpendicular distance from a point to a line."""
#     x0, y0 = point
#     x1, y1 = line_start
#     x2, y2 = line_end
#     num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
#     den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
#     return num / den

# def projection_ratio(point, line_start, line_end):
#     """
#     Compute the projection ratio of a point onto a line segment.
#     Returns a scalar between 0 and 1 representing where the perpendicular
#     projection of the point falls relative to the line segment.
#     """
#     point = np.array(point)
#     line_start = np.array(line_start)
#     line_end = np.array(line_end)
#     line_vec = line_end - line_start
#     if np.linalg.norm(line_vec) == 0:
#         return 0
#     t = np.dot(point - line_start, line_vec) / np.dot(line_vec, line_vec)
#     return t

# def is_new_detection(centroid, detected_list, threshold):
#     """Check if this centroid is far enough from any previously counted centroid."""
#     for pt in detected_list:
#         if np.linalg.norm(np.array(centroid) - np.array(pt)) < threshold:
#             return False
#     return True

# # Open video
# cap = cv2.VideoCapture('/Users/marsman/custom/github/vehicle_counting_tensorflow/traffic4way2.mp4')
# cap.set(3, 1920)
# cap.set(4, 1080)

# if cap.isOpened():
#     ret, frame1 = cap.read()
# else:
#     ret = False

# ret, frame1 = cap.read()
# ret, frame2 = cap.read()

# # Pre-calculate full diagonal line endpoints based on frame size
# frame_width = 1920
# frame_height = 1080
# # Full diagonal from top-left to bottom-right
# diag1_full_start = (0, 0)
# diag1_full_end = (frame_width, frame_height)
# # Full diagonal from top-right to bottom-left
# diag2_full_start = (frame_width, 0)
# diag2_full_end = (0, frame_height)

# # Calculate padded line endpoints (only drawing the middle 60%)
# diag1_draw_start = (int(frame_width * padding_ratio_min), int(frame_height * padding_ratio_min))
# diag1_draw_end   = (int(frame_width * padding_ratio_max), int(frame_height * padding_ratio_max))
# diag2_draw_start = (int(frame_width * (1 - padding_ratio_min)), int(frame_height * padding_ratio_min))
# diag2_draw_end   = (int(frame_width * (1 - padding_ratio_max)), int(frame_height * padding_ratio_max))

# while ret:
#     d = cv2.absdiff(frame1, frame2)
#     grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grey, (5, 5), 0)
#     ret2, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
#     dilated = cv2.dilate(th, np.ones((3, 3)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#     closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Draw the padded diagonal lines on the frame
#     cv2.line(frame1, diag1_draw_start, diag1_draw_end, (0, 255, 0), 2)
#     cv2.line(frame1, diag2_draw_start, diag2_draw_end, (0, 255, 0), 2)
    
#     for c in contours:
#         (x, y, w, h) = cv2.boundingRect(c)
#         # Validate contour size
#         if w < min_contour_width or h < min_contour_height:
#             continue
        
#         # Draw bounding box for visualization
#         cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
#         centroid = get_centroid(x, y, w, h)
#         cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
        
#         # Check distances to the full diagonal lines
#         dist1 = point_line_distance(centroid, diag1_full_start, diag1_full_end)
#         dist2 = point_line_distance(centroid, diag2_full_start, diag2_full_end)
        
#         # Calculate projection ratios on each full line
#         proj_ratio1 = projection_ratio(centroid, diag1_full_start, diag1_full_end)
#         proj_ratio2 = projection_ratio(centroid, diag2_full_start, diag2_full_end)
        
#         # Check if centroid is near a line and its projection falls within the padded region
#         near_diag1 = (dist1 < offset) and (padding_ratio_min <= proj_ratio1 <= padding_ratio_max)
#         near_diag2 = (dist2 < offset) and (padding_ratio_min <= proj_ratio2 <= padding_ratio_max)
        
#         if (near_diag1 or near_diag2) and is_new_detection(centroid, already_counted, threshold=30):
#             vehicles += 1
#             already_counted.append(centroid)
    
#     cv2.putText(frame1, f"Total Vehicles: {vehicles}", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
#     cv2.imshow("Vehicle Detection", frame1)
    
#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
#     # Update frames for next iteration
#     frame1 = frame2
#     ret, frame2 = cap.read()

# cv2.destroyAllWindows()
# cap.release()
