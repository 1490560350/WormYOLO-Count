import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/wormyolo/yolov8m-wormyolo.yaml')
  # model.load('best.pt') # 是否加载预训练权重

    model.train(
                # Train settings -----------------------------------------------------------------------------------------------
                data='ultralytics/cfg/datasets/data.yaml', # (str, optional) path to data file
                epochs=100, # (int) number of epochs to train for
                patience=1000, # (int) epochs to wait for no observable improvement for early stopping of training
                batch=8, # (int) number of images per batch (-1 for AutoBatch)
                imgsz=912, # (int | list) input images size as int for train and val modes, or list[h,w] for predict and export modes
                save=True, # (bool) save train checkpoints and predict results
                save_period=-1, # (int) Save checkpoint every x epochs (disabled if < 1)
                device=0, # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
                workers=8, # (int) number of worker threads for data loading (per RANK if DDP)
                project='runs/train', # (str, optional) project name
                name='exp', # (str, optional) experiment name, results saved to 'project/name' directory
                pretrained=True, # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
                optimizer='Adam', # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
                # Hyperparameters ----------------------------------------------------------------------------------------------
                lr0=1E-3  # (float) initial learning rate
            )
