# WormYOLO(Image segmentation algorithm)
![描述](WormYOLO/examples/Figure1.tif)
### Steps

1.Clone the repository:
   https://github.com/1490560350/WormYOLO-Count.git

2. Setup environments:
   #for running the Deep-Worm-Tracker using pretrained model weights (Quick start)
   cd WormYOLO-Count/WormYOLO
   conda create -n wormyolo
   pip install -e.
   
   #for training yolo model
   cd yolov5
   conda create -n yolo
   pip install -r requirements.txt
   
   #for training torchreid model
   cd strong_sort
   conda create -n torchreid
   pip install -r requirements.txt
3. Running the tracker after cloning the WormYOLO repository:
   python train.py
   python prediction.py
   python tracking.py
# Feature points extraction
### Steps
1. Place the preprocessed image set into the subfolder named "samples" within the "feature points extraction" folder.

2. Double click to open ‘feature points extraction.sln’.

3. Run the program and obtain the coordinates of the pharynx, inflection point, peak point, and skeletal point. These results are saved in the files below.

# Count
### Steps
1. Please place the 5 ".csv" files obtained from the feature point detection algorithm into the "Feature_point" subfolder within the "Tracker" folder.

2. Open the program using Matlab R2021a, run ‘Tracker_ Feature.p’, and obtain the experimental results.



