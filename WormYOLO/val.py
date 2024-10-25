import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('best.pt')
    results=model.val(data=r'CSB-1.yaml',
              split='val',
              imgsz=912,
              batch=12,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
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
    fps = 1000 / total_time
    print(f"FPS: {fps}") # FPS
