import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('ultralytics/cfg/models/v8/yolov8-wormyolo.yaml')
    ##Loading pre-trained models, and need to configure the path correctly.
    #model = YOLO('ultralytics/cfg/models/v8/yolov8-wormyolo.yaml').load('best.pt')

model.train(
    data=r'CSB-1.yaml',
    epochs=150,  # (int) Number of training epochs
    patience=-1,  # (int) Number of epochs to wait for early stopping without improvement
    batch=16,  # (int) Number of images per batch (-1 for auto batching)
    imgsz=912,  # (int) Size of input images, integer or w,h
    save=True,  # (bool) Save training checkpoints and prediction results
    save_period=-1,  # (int) Save checkpoint every x epochs (disable if less than 1)
    cache=False,  # (bool) True/ram, disk, or False. Use cache to load data
    device=0,  # (int | str | list, optional) Device to run on, e.g., cuda device=0 or device=0,1,2,3 or device=cpu
    workers=12,  # (int) Number of worker threads for data loading (per DDP process)
    project='runs/train',  # (str, optional) Project name
    name='exp',  # (str, optional) Experiment name, results saved in 'project/name' directory
    exist_ok=False,  # (bool) Overwrite existing experiment if True
    pretrained=False,  # (bool | str) Use pretrained model (bool), or load weights from model (str)
    optimizer='Adam',  # (str) Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    verbose=True,  # (bool) Print detailed output if True
    seed=0,  # (int) Random seed for reproducibility
    deterministic=True,  # (bool) Enable deterministic mode if True
    single_cls=False,  # (bool) Train multi-class data as a single class
    rect=False,  # (bool) Rectangular training if mode='train', rectangular validation if mode='val'
    cos_lr=True,  # (bool) Use cosine learning rate scheduler
    close_mosaic=0,  # (int) Disable mosaic augmentation for the last few epochs
    resume=True,  # (bool) Resume training from the last checkpoint
    amp=True,  # (bool) Automatic Mixed Precision (AMP) training, choose=[True, False], True runs AMP check
    fraction=1.0,  # (float) Fraction of the dataset to train on (default is 1.0, all images in training set)
    profile=False,  # (bool) Enable ONNX and TensorRT speed profiling during training
    freeze=None,  # (int | list, optional) Freeze the first n layers or list of layer indices during training
    # Segmentation
    overlap_mask=True,  # (bool) Should masks overlap during training (segmentation training only)
    mask_ratio=4,  # (int) Mask downsampling ratio (segmentation training only)
    # Classification
    dropout=0.0,  # (float) Apply dropout regularization (classification training only)
    # Hyperparameters ----------------------------------------------------------------------------------------------
    lr0=1E-3,  # (float) Initial learning rate (e.g., SGD=1E-2, Adam=1E-3)
    lrf=0.01,  # (float) Final learning rate (lr0 * lrf)
    momentum=0.937,  # (float) SGD momentum / Adam beta1
    weight_decay=0.0005,  # (float) Optimizer weight decay 5e-4
    warmup_epochs=3.0,  # (float) Warm-up epochs (fractional values allowed)
    warmup_momentum=0.8,  # (float) Initial momentum during warm-up
    warmup_bias_lr=0.1,  # (float) Initial bias learning rate during warm-up
    box=7.5,  # (float) Box loss gain
    cls=0.5,  # (float) Class loss gain (relative to pixel size)
    dfl=1.5,  # (float) Distribution Focal Loss (DFL) gain
    pose=12.0,  # (float) Pose loss gain
    kobj=1.0,  # (float) Keypoint object loss gain
    label_smoothing=0.0,  # (float) Label smoothing (fractional values)
    nbs=64,  # (int) Nominal batch size
    hsv_h=0.015,  # (float) Image HSV-Hue augmentation (fractional values)
    hsv_s=0.7,  # (float) Image HSV-Saturation augmentation (fractional values)
    hsv_v=0.4,  # (float) Image HSV-Value augmentation (fractional values)
    degrees=0.0,  # (float) Image rotation (+/- degrees)
    translate=0.1,  # (float) Image translation (+/- fraction)
    scale=0.5,  # (float) Image scaling (+/- gain)
    shear=0.0,  # (float) Image shearing (+/- degrees)
    perspective=0.0,  # (float) Image perspective (+/- fraction), range 0-0.001
    flipud=0.0,  # (float) Image vertical flip (probability)
    fliplr=0.5,  # (float) Image horizontal flip (probability)
    mosaic=1.0,  # (float) Image mosaic (probability)
    mixup=0.0,  # (float) Image mixup (probability)
    copy_paste=0.0,  # (float) Segmentation copy-paste (probability)
    auto_augment='randaugment',  # (str) Auto augmentation policy for classification (randaugment, autoaugment, augmix)
    erasing=0.4,  # (float) Probability of random erasing during classification training (0-0.9), 0 means no erasing, must be less than 1.0
    crop_fraction=1.0  # (float) Image crop fraction for classification (0.1-1), 1.0 means no crop, must be greater than 0
)
