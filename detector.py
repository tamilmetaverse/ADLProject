import cv2
import numpy as np
import math
from collections import defaultdict
from ultralytics import YOLO


class PeopleCounter:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.track_history = defaultdict(list)
        self.last_tracked = {}  # Store the last bounding box and ID for each person
        self.people_count = 0
        self.entry_count = 0
        self.last_entry_count = 0  # Store the last displayed entry count
        self.last_total_count = 0  # Store the last displayed total count
        self.current_ids = set()
        self.process_scale = 0.5
        self.skip_frames = 3  # Process every 5th frame
        self.progress = 0
        self.processing = False
        self.completed = False
        self.entry_line = None
        self.frame_count = 0
        self.last_preview_time = 0
        self.preview_interval = 1  # seconds between previews
        self.speeds = {}  # Store the speed of each tracked person

    def set_video_properties(self, width, height):
        x_pos = int(width * 0.3)
        self.entry_line = {
            'x': x_pos,
            'start': (x_pos, 0),
            'end': (x_pos, height),
            'crossed': set()
        }

    def calculate_speed(self, track_id, prev_x, prev_y, curr_x, curr_y, fps):
        # Calculate the distance in pixels
        distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        
        # Convert pixels per frame to km/h
        # Assuming 1 pixel = 0.026 meters (adjust based on your video resolution)
        meters_per_pixel = 0.026
        distance_meters = distance * meters_per_pixel
        speed_mps = distance_meters * fps  # Speed in meters per second
        speed_kmph = speed_mps * 3.6  # Convert to km/h
        return round(speed_kmph, 2)

    def process_frame(self, frame):
        if not self.processing:
            return frame

        height, width = frame.shape[:2]
        if self.entry_line is None:
            self.set_video_properties(width, height)

        # Draw entry line
        cv2.line(frame, self.entry_line['start'], self.entry_line['end'], (0, 0, 255), 2)
        cv2.putText(frame, "Entry", (self.entry_line['x'] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Process every nth frame
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            # Draw last tracked bounding boxes and speeds
            for track_id, box in self.last_tracked.items():
                x1, y1, x2, y2 = box
                speed = self.speeds.get(track_id, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}, Speed: {speed} km/h", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return frame

        small_frame = cv2.resize(frame, None, fx=self.process_scale, fy=self.process_scale)
        results = self.model.track(small_frame, persist=True, classes=[0], verbose=False)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            boxes = boxes / self.process_scale

            # Update current IDs
            self.current_ids = set(track_ids)

            # Track people in the entry zone
            people_in_entry_zone = set()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Calculate speed
                if track_id in self.last_tracked:
                    prev_x1, prev_y1, prev_x2, prev_y2 = self.last_tracked[track_id]
                    prev_center_x = (prev_x1 + prev_x2) // 2
                    prev_center_y = (prev_y1 + prev_y2) // 2
                    self.speeds[track_id] = self.calculate_speed(
                        track_id, prev_center_x, prev_center_y, center_x, center_y, fps=30  # Adjust FPS as needed
                    )

                # Update last tracked information
                self.last_tracked[track_id] = (x1, y1, x2, y2)

                # Check if the person is in the entry zone
                if center_x > self.entry_line['x']:
                    people_in_entry_zone.add(track_id)

                # Draw bounding box and speed
                speed = self.speeds.get(track_id, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}, Speed: {speed} km/h", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update entry count
            self.entry_count = len(people_in_entry_zone)

        # Remove stale entries from last_tracked
        self.last_tracked = {track_id: box for track_id, box in self.last_tracked.items() if track_id in self.current_ids}

        # Display counts
        cv2.putText(frame, f"Total People: {len(self.current_ids)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Entry Count: {self.entry_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame