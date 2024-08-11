import os
import shutil
import cv2
import numpy as np
from scipy.spatial import distance
from mmdet.apis import init_detector, inference_detector
from PIL import Image

# Segmentation functions
def calculate_black_area(mask):
    return np.sum(mask)

def process_image(model, img_folder, result_folder, filename, confidence_threshold, num_targets):
    img_path = os.path.join(img_folder, filename)
    result = inference_detector(model, img_path)
    masks = result[1][0][:num_targets]
    scores = result[0][0][:num_targets, 4] if result[0][0].shape[1] > 4 else np.ones(len(masks))

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
        save_path = os.path.join(result_folder, f'{os.path.splitext(filename)[0]}_{target_index}.png')
        img_kou.save(save_path)

# Classification functions
def calculate_center_of_black_area(image):
    black_pixels = np.where(image == 0)
    if len(black_pixels[0]) == 0:
        return None
    x_min, x_max = min(black_pixels[1]), max(black_pixels[1])
    y_min, y_max = min(black_pixels[0]), max(black_pixels[0])
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    return center


def find_closest_centers(center, centers, max_dist=15):
    close_centers = []
    for img_name, img_center in centers.items():
        dist = distance.euclidean(center, img_center)
        if dist < max_dist:
            close_centers.append((img_name, img_center, dist))
    return close_centers


def find_most_overlapping_image(current_image, close_centers, input_folder):
    max_overlap = 0
    best_image = None
    current_image_path = os.path.join(input_folder, current_image)
    current_img = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    current_black_area = (current_img == 0).astype(np.uint8)
    for img_name, _, _ in close_centers:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        black_area = (img == 0).astype(np.uint8)
        overlap = np.sum(current_black_area & black_area)
        if overlap > max_overlap:
            max_overlap = overlap
            best_image = img_name
    return best_image


def process_images(start_image, images, input_folder, centers_cache, max_dist=15):
    prefix = int(start_image.split('_')[0])
    nxtprefix = prefix + 1
    current_image = start_image
    found_images = []
    distances = []

    while True:
        current_image_path = os.path.join(input_folder, current_image)
        image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        current_center = calculate_center_of_black_area(image)
        if current_center is None:
            break
        found_images.append(current_image)
        next_prefix_images = {f: centers_cache[f] for f in images if
                              f.startswith(f"{nxtprefix}_") and centers_cache[f] is not None}
        if not next_prefix_images:
            break
        close_centers = find_closest_centers(current_center, next_prefix_images, max_dist)
        if not close_centers:
            found_images.pop()
            nxtprefix += 1
            continue
        if len(close_centers) == 1:
            closest_image, _, min_dist = close_centers[0]
        else:
            closest_image = find_most_overlapping_image(current_image, close_centers, input_folder)
            min_dist = min([dist for _, _, dist in close_centers])
        distances.append(min_dist)
        avg_distance = np.mean(distances)
        current_image = closest_image
        nxtprefix += 1

    return found_images

def run_segmentation(model, img_folder, result_folder, confidence_threshold, num_targets):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for filename in os.listdir(img_folder):
        if filename.endswith('.png'):
            process_image(model, img_folder, result_folder, filename, confidence_threshold=confidence_threshold,
                          num_targets=num_targets)

def prepare_classification(segmentation_folder, all_folder):
    if not os.path.exists(all_folder):
        os.makedirs(all_folder)

    images = [f for f in os.listdir(segmentation_folder) if f.endswith(".png")]
    for img in images:
        src_path = os.path.join(segmentation_folder, img)
        dest_path = os.path.join(all_folder, img)
        shutil.copy(src_path, dest_path)


def run_classification(all_folder, result_folder):
    images = sorted([f for f in os.listdir(all_folder) if f.endswith(".png")], key=lambda x: int(x.split('_')[0]))
    centers_cache = {f: calculate_center_of_black_area(cv2.imread(os.path.join(all_folder, f), cv2.IMREAD_GRAYSCALE))
                     for f in images}
    processed_images = set()

    for start_image in images:
        if start_image in processed_images:
            continue
        found_images = process_images(start_image, images, all_folder, centers_cache)
        output_folder = os.path.join(result_folder, "worm" + start_image.split('_')[1].split('.')[0])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for img in found_images:
            processed_images.add(img)
            src_path = os.path.join(all_folder, img)
            dest_path = os.path.join(output_folder, img)
            if not os.path.exists(dest_path):
                shutil.move(src_path, dest_path)
            images = [f for f in os.listdir(all_folder) if f.endswith(".png")]
            centers_cache = {f: centers_cache[f] for f in images}

    return processed_images


def cleanup(all_folder):
    if os.path.exists(all_folder):
        shutil.rmtree(all_folder)

def main():
    config_file = 'worm_inference.py'
    checkpoint_file = 'multi_worm.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_folder = 'worm_data'
    result_folder = os.path.join(img_folder, 'result')

    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.makedirs(result_folder)

    segmentation_folder = result_folder
    all_folder = os.path.join(result_folder, "all")

    # Set the confidence threshold for segmentation(The default threshold is set to 0. If the results are not accurate, consider adjusting the threshold.)
    confidence_threshold = 0.0
    num_targets = None
    run_segmentation(model, img_folder, segmentation_folder, confidence_threshold, num_targets)
    prepare_classification(segmentation_folder, all_folder)
    run_classification(all_folder, result_folder)
    cleanup(all_folder)

if __name__ == "__main__":
    main()
