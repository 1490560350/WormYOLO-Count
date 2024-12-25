from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
import random
import logging


# 初始化YOLO模型
model = YOLO("weight/CSB-1.pt")

# 输入视频路径
video_path = "video/24_2_1_1.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    logging.error("无法打开视频文件")
    exit()

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 提取视频文件名（不含扩展名）
video_filename = os.path.splitext(os.path.basename(video_path))[0]

# 创建主输出目录
main_output_dir = os.path.join("track_result", video_filename)
os.makedirs(main_output_dir, exist_ok=True)

# 初始化综合视频编写器
all_masks_path = os.path.join(main_output_dir, "track_video.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
all_masks_writer = cv2.VideoWriter(all_masks_path, fourcc, fps, (width, height))

# 初始化txt文件路径
txt_file_path = os.path.join(main_output_dir, "track_boxes.txt")
with open(txt_file_path, "w") as f:
    f.write("")  # 清空文件内容

frame_count = 0  # 用于跟踪处理的帧数

# 用于存储track_id对应的颜色
track_id_colors = {}

def generate_random_color():
    """生成随机颜色"""
    return tuple(random.randint(0, 255) for _ in range(3))

# 跳过帧数的设置
frame_skip = 1  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue  # 跳过处理

    try:
        # 进行对象跟
        results = model.track(frame, persist=True,conf=0.3, tracker="botsort.yaml")

        # 获取检测到的框和对应的跟踪ID
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取左上角和右下角坐标
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # 获取掩码坐标
        masks = results[0].masks.xy

        # 创建一个字典，键为track_id，值为对应的掩码
        mask_dict = defaultdict(list)
        if masks is not None:
            for mask, track_id in zip(masks, track_ids):
                mask_dict[track_id].append(mask)

        # 遍历所有track_id，输出掩码图像
        for track_id in mask_dict:
            track_output_dir = os.path.join(main_output_dir, f"id{track_id}_image")
            os.makedirs(track_output_dir, exist_ok=True)

            mask_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

            for idx, mask in enumerate(mask_dict[track_id]):
                mask_points = mask.reshape((-1, 1, 2)).astype(np.int32)
                if mask_points.size == 0:
                    logging.warning(f"Track ID {track_id} 在第 {frame_count} 帧的掩码为空。")
                    continue
                cv2.fillPoly(mask_frame, [mask_points], (0, 0, 0))  # 黑色

                output_image_path = os.path.join(track_output_dir, f"{frame_count-1}.png")
                cv2.imwrite(output_image_path, mask_frame)

        # 在原始帧上绘制检测框、ID 和掩码
        all_masks_frame = frame.copy()

        if masks is not None:
            for bbox, track_id, mask in zip(boxes, track_ids, masks):
                if track_id not in track_id_colors:
                    track_id_colors[track_id] = generate_random_color()

                color = track_id_colors[track_id]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(all_masks_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(all_masks_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                mask_points = mask.reshape((-1, 1, 2)).astype(np.int32)
                if mask_points.size == 0:
                    logging.warning(f"Track ID {track_id} 在第 {frame_count} 帧的掩码为空。")
                    continue
                single_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(single_mask, [mask_points], 255)
                color_mask = np.zeros_like(frame)
                color_mask[:, :] = color
                colored_mask = cv2.bitwise_and(color_mask, color_mask, mask=single_mask)
                all_masks_frame = cv2.addWeighted(all_masks_frame, 1.0, colored_mask, 0.2, 0)

                with open(txt_file_path, "a") as f:
                    f.write(f"{frame_count-1},{track_id},{x1-3},{y1+3},{x2 - x1+3},{y2 - y1+3},-1,-1,-1\n")

        all_masks_writer.write(all_masks_frame)

    except Exception as e:
        logging.error(f"在第 {frame_count} 帧处理时出错: {e}")
        continue

cap.release()
all_masks_writer.release()


logging.info(f"追踪结果已保存到 {main_output_dir} 目录下")
