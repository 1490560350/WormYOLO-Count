import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('weight/synthetic.pt')
    results=model.val(
              data='ultralytics/cfg/datasets/data.yaml',
              split='val',
              imgsz=912,
              batch=8,
              project='runs/val',
              name='exp',
              )
    
    print(f"box-mAP50: {results.box.map50}")  # map50
    print(f"box-mAP75: {results.box.map75}")  # map75
    print(f"box-mAP50-95: {results.box.map}") # map50-95
   
    print(f"mask-mAP50: {results.seg.map50}")  # map50
    print(f"mask-mAP75: {results.seg.map75}")  # map75
    print(f"mask-mAP50-95: {results.seg.map}") # map50-95
    
    speed_results = results.speed
    total_time = sum(speed_results.values())
    fps = 1000 / total_time #
    print(f"FPS: {fps}") # FPS
