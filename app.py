import os
import time
import threading
from flask import Flask, render_template, jsonify, request, send_file
from werkzeug.utils import secure_filename
from detector import PeopleCounter  # Import PeopleCounter
import cv2
import webbrowser

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure paths
base_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(base_dir, 'outputs')
app.config['PREVIEW_FOLDER'] = os.path.join(base_dir, 'static', 'previews')

# Create required directories
for directory in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['PREVIEW_FOLDER']]:
    os.makedirs(directory, exist_ok=True)

# Create placeholder image
placeholder_path = os.path.join(app.config['PREVIEW_FOLDER'], 'placeholder.jpg')
if not os.path.exists(placeholder_path):
    cv2.imwrite(placeholder_path, np.zeros((100, 100, 3), dtype=np.uint8))

# Initialize PeopleCounter
counter = PeopleCounter()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{filename}")
        
        file.save(input_path)
        
        # Reset counter for new video
        counter.__init__()
        counter.processing = True
        
        thread = threading.Thread(
            target=process_video,
            args=(input_path, output_path, timestamp),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'output': f"processed_{filename}",
            'timestamp': timestamp
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video(input_path, output_path, timestamp):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video: {input_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        counter.set_video_properties(width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened() and counter.processing:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = counter.process_frame(frame)
            out.write(processed_frame)
            
            # Save preview at regular intervals
            current_time = time.time()
            if current_time - counter.last_preview_time > counter.preview_interval:
                preview_path = os.path.join(app.config['PREVIEW_FOLDER'], f"preview_{timestamp}.jpg")
                cv2.imwrite(preview_path, processed_frame)
                counter.last_preview_time = current_time
            
            frame_count += 1
            counter.progress = int((frame_count / total_frames) * 100)
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({counter.progress}%)")
        
        cap.release()
        out.release()
        counter.processing = False
        counter.completed = True
        
    except Exception as e:
        print(f"Processing error: {e}")
        counter.processing = False
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/progress')
def get_progress():
    return jsonify({
        'progress': counter.progress,
        'processing': counter.processing,
        'completed': counter.completed,
        'entry_count': counter.entry_count,
        'total_count': len(counter.current_ids)
    })

@app.route('/preview/<timestamp>')
def get_preview(timestamp):
    preview_path = os.path.join(app.config['PREVIEW_FOLDER'], f"preview_{timestamp}.jpg")
    if os.path.exists(preview_path):
        return send_file(preview_path, mimetype='image/jpeg')
    return send_file(placeholder_path, mimetype='image/jpeg')

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

import time  # Import time for delay

@app.route('/exit', methods=['POST'])
def exit_app():
    # Render the thank you page first
    response = render_template('thank_you.html')
    
    # Delay the shutdown to allow the browser to render the page
    shutdown = request.environ.get('werkzeug.server.shutdown')
    if shutdown is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    
    # Use threading to delay the shutdown
    threading.Thread(target=lambda: (time.sleep(2), shutdown())).start()
    
    return response

if __name__ == '__main__':
    # Open browser automatically
    webbrowser.open('http://localhost:5000')
    
    # Start Flask app
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    app.run(host='0.0.0.0', port=5000, threaded=True)