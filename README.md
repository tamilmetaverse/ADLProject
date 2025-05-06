# Yolov8-object-detection-and-tracking-region-wise
Flask web app for real-time YOLOv8 object detection, tracking, region-wise counting, and speed estimation in videos. Includes Docker deployment and sample videos.



# YOLOv8 Object Detection, Tracking, Region Counting, and Speed Estimation

This project implements a Flask web application that uses a YOLOv8 model for real-time object detection, tracking, region-wise counting, and speed estimation in videos.

---

## ✨ Features

- Object detection and tracking (YOLOv8)
- Region-based people counting (entry line)
- Speed estimation in km/h
- Flask web UI for video upload and visualization
- Dockerized for easy deployment

---

## Project Structure📁

```plaintext
project_root/
├── app.py                # Main Flask application file
├── detector.py           # YOLOv8 object detection and tracking module
├── yolov8n.pt            # Pretrained YOLOv8 model weights
├── requirements.txt      # Python dependencies list
├── Dockerfile            # Docker configuration file
├── static/               # Static files directory
│   └── sample_videos/    # Sample videos for testing
└── templates/            # HTML templates directory
    └── index.html        # Main web UI template
```

## Conclusion:

The system does a great job at detecting and tracking people in real-time. It shows green bounding boxes around each person, displays their unique IDs and walking speeds, and keeps track of both the entry count and the total number of people on screen. Overall, it works smoothly and achieves the main goals. That said, it doesn’t yet handle person re-identification, and it faces some of the usual challenges you see in object detection and tracking, like occasional missed detections or ID switches when people overlap or move fast. With some improvements in these areas, it could become even more robust and accurate.

### Limitations:

- No person re-identification across long gaps or when people leave and re-enter the frame.
- Occasional ID switches, especially in crowded or fast-moving scenes.
- Missed detections in low-light, blurry, or occluded situations.
- Speed estimation may vary slightly depending on camera angle and video quality.


## Future Work:


- Add a person re-identification module to maintain consistent IDs across frames, even after occlusion or re-entry.
- Improve tracking robustness in crowded scenes with better multi-object tracking algorithms.
- Fine-tune the detection model on more diverse datasets to handle challenging conditions like low light or occlusion.
- Enhance the speed estimation logic using camera calibration for better accuracy.
- Expand the web app with features like exporting reports, real-time alerts, or multi-camera support.

