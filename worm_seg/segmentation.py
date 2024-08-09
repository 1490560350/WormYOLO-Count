import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
from PIL import Image

def calculate_black_area(mask):
    return np.sum(mask)

def process_image(model, img_folder, output_folder, filename, confidence_threshold=0.0, num_targets=5):
    img_path = os.path.join(img_folder, filename)
    result = inference_detector(model, img_path)
    masks = result[1][0][:num_targets]
    scores = result[0][0][:num_targets, 4] if result[0][0].shape[1] > 4 else np.ones(len(masks)) 
    print(f"Processed document: {filename}")

    target_index = 0
    skipped_masks = 0
    black_areas = []
    retained_areas = []

    for i, mask in enumerate(masks):
        black_area = calculate_black_area(mask)

        if confidence_threshold > 0.5:  
            if scores[i] >= confidence_threshold:
                retained_areas.append(black_area)
            else:
                skipped_masks += 1
                print(f"Skipped mask {i + 1} with confidence score: {scores[i]}")
                if skipped_masks >= 3:
                    break
                continue
        else:  
            if target_index == 0:
                black_areas.append(black_area)
                retained_areas.append(black_area)
            else:
                if len(black_areas) > 0:
                    avg_black_area = np.mean(black_areas)
                    if black_area >= 0.6 * avg_black_area:
                        black_areas.append(black_area)
                        retained_areas.append(black_area)
                    else:
                        skipped_masks += 1
                        if skipped_masks >= 3:
                            break
                        continue

        target_index += 1
        np_img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255
        np_img[mask] = [0, 0, 0]
        img_kou = Image.fromarray(np_img)
        save_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_{target_index}.png')
        img_kou.save(save_path)


def run_test_script(batch_size=10, num_targets=5, confidence_threshold=0.999):
    config_file = 'worm_inference.py'
    checkpoint_file = 'multi_worm.pth'
    model = init_detector(config_file, checkpoint_file)
    img_folder = os.path.join(os.path.dirname(__file__), 'worm_data')  
    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"The folder {img_folder} does not exist")

    output_folder = os.path.join(img_folder, 'results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i:i + batch_size]
        for filename in batch_files:
            process_image(model, img_folder, output_folder, filename, confidence_threshold, num_targets)


if __name__ == "__main__":
    run_test_script(batch_size=20, confidence_threshold=0)
