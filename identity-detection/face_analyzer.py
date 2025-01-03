import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import time
from collections import deque
import threading
from queue import Queue
import os
import sys
from datetime import datetime
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

class FaceAnalyzer:
    def __init__(self, verbose=False, buffer_size=15, update_interval=1.0, 
                 min_votes=5, max_display=8, confidence_threshold=0.6):
        self.verbose = verbose
        self.analysis_queue = Queue(maxsize=1)
        self.analysis_thread = None
        self.running = True
        self.current_analysis = None
        self.last_results = []
        self.max_display_results = max_display
        
        # Configurable buffer sizes
        self.age_buffer = deque(maxlen=buffer_size)
        self.gender_votes = deque(maxlen=buffer_size)
        self.race_votes = deque(maxlen=buffer_size)
        
        self.min_votes_needed = min_votes
        self.last_update_time = 0
        self.update_interval = update_interval
        self.confidence_threshold = confidence_threshold
        
        # Configure logging
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        
        if self.verbose:
            print("\n📊 Loading AI Models...")
            self._print_settings()
        
        _ = DeepFace.analyze(
            np.zeros((100, 100, 3), dtype=np.uint8),
            actions=['age', 'gender', 'race'],
            enforce_detection=False,
            silent=True
        )
        
        if self.verbose:
            print("✅ Models loaded successfully!")
        
        self.start_analysis_thread()
    
    def _print_settings(self):
        print("\nCurrent Settings:")
        print(f"├── Buffer Size: {self.age_buffer.maxlen}")
        print(f"├── Update Interval: {self.update_interval}s")
        print(f"├── Minimum Votes: {self.min_votes_needed}")
        print(f"├── Max Display Results: {self.max_display_results}")
        print(f"└── Confidence Threshold: {self.confidence_threshold}\n")

    # ... [rest of the FaceAnalyzer class methods remain the same] ...

def main():
    # Suppress all warnings
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    
    # Set up argument parser with more configuration options
    parser = argparse.ArgumentParser(
        description='Face Analysis System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('-b', '--buffer-size',
                       type=int,
                       default=15,
                       help='Size of the rolling buffer for smoothing results')
    
    parser.add_argument('-u', '--update-interval',
                       type=float,
                       default=1.0,
                       help='Interval between updates in seconds')
    
    parser.add_argument('-m', '--min-votes',
                       type=int,
                       default=5,
                       help='Minimum votes needed before showing results')
    
    parser.add_argument('-d', '--display-results',
                       type=int,
                       default=8,
                       help='Number of results to display in history')
    
    parser.add_argument('-c', '--confidence',
                       type=float,
                       default=0.6,
                       help='Confidence threshold for detection (0.0-1.0)')
    
    parser.add_argument('--width',
                       type=int,
                       default=640,
                       help='Camera width resolution')
    
    parser.add_argument('--height',
                       type=int,
                       default=480,
                       help='Camera height resolution')
    
    parser.add_argument('--fps',
                       type=int,
                       default=30,
                       help='Camera FPS')
    
    args = parser.parse_args()

    if args.verbose:
        print("\n🚀 Initializing Face Analysis System...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if args.verbose:
        print("📷 Loading YOLO model...")
    
    # Initialize YOLO model with verbosity settings
    model = YOLO('yolov8n.pt')
    model.conf = args.confidence
    
    face_analyzer = FaceAnalyzer(
        verbose=args.verbose,
        buffer_size=args.buffer_size,
        update_interval=args.update_interval,
        min_votes=args.min_votes,
        max_display=args.display_results,
        confidence_threshold=args.confidence
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Run detection with suppressed output
            if not args.verbose:
                with open(os.devnull, 'w') as devnull:
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    results = model(frame, verbose=False)
                    sys.stdout = old_stdout
            else:
                results = model(frame, verbose=True)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    conf = float(box.conf[0])
                    
                    if class_name == 'person' and conf > args.confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        body_height = y2 - y1
                        face_height = int(body_height * 0.3)
                        face_y2 = y1 + face_height
                        
                        if face_analyzer.analysis_queue.empty():
                            face_analyzer.analysis_queue.put((frame, (x1, y1, x2, face_y2), conf))
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        if args.verbose:
            print("\n👋 Stopping analysis...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        face_analyzer.stop()
        cap.release()

if __name__ == "__main__":
    main() 