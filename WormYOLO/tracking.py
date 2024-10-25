import warnings
warnings.filterwarnings('ignore')
import cv2, os, shutil
import numpy as np
from ultralytics import YOLO
from boxmot import DeepOcSort
from pathlib import Path

def get_video_cfg(path):
    video = cv2.VideoCapture(path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter_fourcc(*'mp4v'), size, fps

def plot_and_counting(frame, tracker):
    # Update the tracker with new detections and get results
    res = tracker.update(dets, frame)  # M X (x1, y1, x2, y2, id, conf, cls, ind)
    
    # Plot tracking results on the image
    tracker.plot_results(frame, show_trajectories=True)
    
    return frame

if __name__ == '__main__':
    output_dir = 'result'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your local YOLO model
    model = YOLO('best.pt').to('cpu')  # Replace with your local model path
    
    # Initialize the tracker
    tracker = DeepOcSort(
          reid_weights=Path('osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pt'), # which ReID model to use
          device="cpu",
           half=False,
    )
       
    
    # ----------------------for video-folder----------------------
    video_base_path = ''
    for video_path in os.listdir(video_base_path):
        fourcc, size, fps = get_video_cfg(f'{video_base_path}/{video_path}')
        video_output = cv2.VideoWriter(f'{output_dir}/{video_path}', fourcc, fps, size)
        
        vid = cv2.VideoCapture(f'{video_base_path}/{video_path}')
        
        while True:
            # Capture frame-by-frame
            ret, frame = vid.read()
            if not ret:
                break
            
            # Perform detection using YOLO
            results = model.predict(frame, stream=True, imgsz=928)
            
            dets = []
            for result in results:
                if result.boxes is not None:
                    for i in range(result.boxes.shape[0]):
                        bbox = result.boxes.xyxy[i].cpu().numpy()
                        conf = result.boxes.conf[i].item()
                        cls = result.boxes.cls[i].item()
                        dets.append([*bbox, conf, cls])  # [x1, y1, x2, y2, conf, cls]
            
            # Convert detections to numpy array (N X (x1, y1, x2, y2, conf, cls))
            dets = np.array(dets)
            
            # Update tracker and plot results
            image_plot = plot_and_counting(frame, tracker)
            video_output.write(image_plot)
        
        # Release resources
        vid.release()
        video_output.release()
    
    cv2.destroyAllWindows()