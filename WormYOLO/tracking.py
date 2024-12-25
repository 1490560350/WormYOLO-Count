from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch

# ------------------------------
# Configuration Variables
# ------------------------------
DRAW_MASK = True      # Set to True to draw mask outlines and semi-transparent fills
DRAW_BOX = True       # Set to True to draw bounding boxes and IDs
EXPAND_PIXELS = 50    # Number of pixels to expand each side of the bounding box
RED_COLOR = [0, 0, 255]  # Red color in BGR format

# Define helper functions
def xyxy2xywh(x):
    """
    Converts bounding boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height].
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]        # width
    y[:, 3] = x[:, 3] - x[:, 1]        # height
    return y

def xywh2xyxy(x):
    """
    Converts bounding boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2].
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def expand_bounding_box(x1, y1, x2, y2, expand_pixels, frame_width, frame_height):
    """
    Expands the bounding box by a specified number of pixels on each side.
    Ensures that the expanded bounding box does not exceed the frame boundaries.

    Parameters:
    - x1, y1, x2, y2 (int): Original bounding box coordinates.
    - expand_pixels (int): Number of pixels to expand on each side.
    - frame_width (int): Width of the frame.
    - frame_height (int): Height of the frame.

    Returns:
    - expanded_x1, expanded_y1, expanded_x2, expanded_y2 (int): Expanded bounding box coordinates.
    """
    expanded_x1 = max(x1 - expand_pixels, 0)
    expanded_y1 = max(y1 - expand_pixels, 0)
    expanded_x2 = min(x2 + expand_pixels, frame_width - 1)
    expanded_y2 = min(y2 + expand_pixels, frame_height - 1)
    return expanded_x1, expanded_y1, expanded_x2, expanded_y2

def process_worm(bboxes, ims, BGR=False, alpha=0.4, color=[0, 0, 255], do_segment=False):
    """
    Processes the detected worm by extracting the ROI, applying thresholding and morphological operations.
    Optionally segments the ROI with a colored mask.

    Parameters:
    - bboxes (numpy.ndarray): Bounding box coordinates in [x1, y1, x2, y2] format.
    - ims (numpy.ndarray): The original image/frame.
    - BGR (bool): Flag indicating if the image is in BGR format.
    - alpha (float): Transparency factor for blending masks.
    - color (list): BGR color for the mask overlay.
    - do_segment (bool): Flag to apply mask segmentation (color overlay).

    Returns:
    - processed_mask (numpy.ndarray or None): The processed mask after morphological operations.
    """
    if bboxes.size == 0:
        return None  # No bounding boxes to process

    bboxes = torch.tensor(bboxes).view(-1,4)  # bbox coordinates
    b = xyxy2xywh(bboxes)  # convert to xywh
    xyxy = xywh2xyxy(b).long()

    # Extract coordinates
    x1, y1, x2, y2 = xyxy[0]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, ims.shape[1]-1), min(y2, ims.shape[0]-1)

    # Extract ROI with correct channel order
    roi = ims[int(y1):int(y2), int(x1):int(x2), ::-1 if not BGR else 1]

    # Convert to grayscale and apply Gaussian Blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0)        # smoothing

    # Apply thresholding and invert
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # thresholding
    invert = cv2.bitwise_not(thresh)               # invert to get black background

    # Find contours
    contours, hierarchy = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    if not contours:
        return None  # No contours found

    # Create mask and draw largest contour
    mask = np.zeros_like(gray)  # create single-channel mask
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1, cv2.LINE_AA)  # draw filled largest contour

    # Define structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # You can adjust the size as needed

    # Apply Dilation
    dilated_mask = cv2.dilate(mask, kernel, iterations=3)

    # Apply Erosion on the Dilated Mask
    processed_mask = cv2.erode(dilated_mask, kernel, iterations=3)

    # Optionally segment the ROI with the processed mask
    if do_segment:
        colored_mask = np.zeros_like(roi)
        colored_mask[processed_mask == 255] = color  # Apply the color
        roi = cv2.addWeighted(roi, alpha, colored_mask, 1 - alpha, 0)  # blend masks with ROI

    # Place the modified ROI back into the original image
    ims[int(y1):int(y2), int(x1):int(x2), ::-1 if not BGR else 1] = roi

    return processed_mask  # Return the processed mask for further visualization

# ------------------------------
# Initialize YOLO Model
# ------------------------------
model_path = "weight/Singleworm.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"YOLO model not found at {model_path}. Please check the path.")
model = YOLO(model_path)

# ------------------------------
# Setup Video Capture and Writer
# ------------------------------
video_path = "video/video1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频文件名（不含扩展名）
video_filename = os.path.splitext(os.path.basename(video_path))[0]

# 创建与视频名字相同的输出子文件夹
output_dir = "track_result"
video_output_dir = os.path.join(output_dir, video_filename)
os.makedirs(video_output_dir, exist_ok=True)

# 定义输出视频路径
output_video_path = os.path.join(video_output_dir, "track_video.avi")

# 定义掩膜图片保存路径
mask_image_dir = os.path.join(video_output_dir, "id1_image")
os.makedirs(mask_image_dir, exist_ok=True)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# ------------------------------
# Initial Frame Processing
# ------------------------------
num_initial_frames = 5
initial_frames = []
for i in range(num_initial_frames):
    success, frame = cap.read()
    if not success:
        print(f"视频前 {i} 帧无法读取。")
        break
    initial_frames.append(frame)

if len(initial_frames) == 0:
    print("无法读取任何帧来计算平均像素值。")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    exit()

# Calculate the average pixel value of the first 5 frames
gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in initial_frames]
avg_pixel_values = [frame.mean() for frame in gray_frames]
overall_avg_pixel = np.mean(avg_pixel_values)

# Decide whether to process frames based on average pixel value
process_frames = False
if overall_avg_pixel < 10:
    process_frames = True
else:
    process_frames = False  # 如果背景不为黑色，设置不处理帧


# Reset video to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ------------------------------
# Define a function to write bounding box information to a text file
# ------------------------------
def write_bounding_boxes_to_text(frame_count, track_ids, boxes, output_txt_file):
    """
    Writes bounding box information to a text file in the following format:
    frame_count, track_id, x1, y1, width, height, -1, -1, -1
    """
    with open(output_txt_file, 'a') as f:
        for idx, box in enumerate(boxes):
            # Extract coordinates and track id
            x1, y1, x2, y2 = map(int, box)
            track_id = track_ids[idx]
            width = x2 - x1
            height = y2 - y1
            
            # Write the formatted string to the file
            f.write(f"{frame_count-1},{track_id},{x1},{y1},{width},{height},-1,-1,-1\n")

# ------------------------------
# Setup text file to save bounding box data
# ------------------------------
txt_file_path = os.path.join(video_output_dir, "track_boxes.txt")

# Clear the file if it exists (start fresh)
if os.path.exists(txt_file_path):
    os.remove(txt_file_path)


# ------------------------------
# Main Processing Loop
# ------------------------------
frame_count = 0  # To track frame numbers
mask_count = 0   # To track mask image numbering

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # 如果需要，替换像素值
    if process_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        origin_frame = frame.copy()
        mask_low = gray < 10
        frame[mask_low] = 125  # Set all channels to 125 where mask is True

    # Perform object tracking using YOLO
    results = model.track(frame, persist=True, retina_masks=True, show_boxes=False, tracker="botsort.yaml")

    # Extract bounding boxes and track IDs
    boxes_xywh = results[0].boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
    track_ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()

    # Convert xywh to xyxy
    if len(boxes_xywh) > 0:
        boxes = xywh2xyxy(torch.tensor(boxes_xywh)).numpy()
    else:
        boxes = np.array([]).reshape(0,4)

    # Write bounding box information to the text file
    write_bounding_boxes_to_text(frame_count, track_ids, boxes, txt_file_path)

    # Initialize a full-frame mask
    full_mask = np.zeros((height, width), dtype=np.uint8)

    # Process each bounding box and collect masks
    annotated_frame = frame.copy()
    for idx, box in enumerate(boxes):
        # Extract original bounding box coordinates
        orig_x1, orig_y1, orig_x2, orig_y2 = box
        orig_x1, orig_y1, orig_x2, orig_y2 = int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)

        # Ensure coordinates are within image boundaries
        orig_x1, orig_y1 = max(orig_x1, 0), max(orig_y1, 0)
        orig_x2, orig_y2 = min(orig_x2, frame.shape[1]-1), min(orig_y2, frame.shape[0]-1)

        # Expand the bounding box by EXPAND_PIXELS on each side
        expanded_x1, expanded_y1, expanded_x2, expanded_y2 = expand_bounding_box(
            orig_x1, orig_y1, orig_x2, orig_y2, EXPAND_PIXELS, frame.shape[1], frame.shape[0]
        )

        # 检查扩展后的边界框是否有效
        if expanded_x1 >= expanded_x2 or expanded_y1 >= expanded_y2:
            continue

        # Prepare the expanded box for processing
        expanded_box_tensor = np.array([[expanded_x1, expanded_y1, expanded_x2, expanded_y2]])

        # Call process_worm and get the processed mask
        # Set do_segment=False to prevent color overlay inside bounding boxes
        mask = process_worm(expanded_box_tensor, annotated_frame, BGR=False, alpha=0.4, 
                           color=RED_COLOR, do_segment=False)

        if mask is not None and mask.any():
            region = full_mask[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

            # 检查尺寸是否匹配
            if mask.shape == region.shape:
                # 确保数据类型一致
                if full_mask.dtype != mask.dtype:
                    mask = mask.astype(full_mask.dtype)
                
                # 进行位运算
                full_mask[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = cv2.bitwise_or(
                    region, mask)
            else:
                mask_resized = cv2.resize(mask, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
                full_mask[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = cv2.bitwise_or(
                    region, mask_resized)

            # **Draw Bounding Boxes and Track IDs in Red**
            if DRAW_BOX:
                # Draw bounding box based on the original (non-expanded) coordinates
                cv2.rectangle(origin_frame, (orig_x1, orig_y1), (orig_x2, orig_y2), RED_COLOR, 2)
                cv2.putText(origin_frame, f'ID: {track_ids[idx]}', 
                            (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, RED_COLOR, 2)

    # 将全帧掩膜转换为黑色前景和白色背景
    # 即掩膜区域为黑色 (0), 其余为白色 (255)
    binary_mask = np.ones_like(full_mask) * 255  # Start with white background
    binary_mask[full_mask == 255] = 0           # Set mask regions to black

    # 保存掩膜图片
    mask_image_path = os.path.join(mask_image_dir, f"{mask_count}.png")
    cv2.imwrite(mask_image_path, binary_mask)
    mask_count += 1

    # 创建一个红色掩膜图层
    red_mask = np.zeros_like(frame, dtype=np.uint8)
    red_mask[binary_mask == 0] = RED_COLOR  # Set mask regions to red

    # 创建半透明效果
    alpha = 0.4  # Transparency factor
    origin_frame = cv2.addWeighted(origin_frame, 1.0, red_mask, alpha, 0)

    # 写入注释后的帧到输出视频
    out.write(origin_frame)

# ------------------------------
# Release Resources
# ------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"追踪结果已保存到 {output_dir}文件夹")
