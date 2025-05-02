import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

IMG_SIZE = 512
NUM_WORKERS = 32  # according to your CPU cores
FUNDUS_PATH = glob(
    "/data0/wc_data/datasets/DDR-dataset/lesion_segmentation/*/image/*.jpg"  # set your own path
)
MASK_PATH = lambda file_name: glob(  # noqa: E731
    f"/data0/wc_data/datasets/DDR-dataset/lesion_segmentation/*/label/*/{file_name}.tif"
)  # set your own path
SAVE_DIR = "data/DDR"  # set your own path


def pad_to_square(img):
    """
    Pads an image to make it a square by adding zeros to the shorter side.
    Args:
        img: PIL Image object.
    Returns:
        A PIL Image object representing a square image.
    """
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        pad = (width - height) // 2
        return ImageOps.expand(img, border=(0, pad, 0, pad), fill=0)
    else:
        pad = (height - width) // 2
        return ImageOps.expand(img, border=(pad, 0, pad, 0), fill=0)


def border_crop_square_padding(img_path: str, mask_imgs):
    # 读取图像
    image = cv2.imread(img_path, 0)

    # Otsu's thresholding
    ret2, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 检测二值图最左边的白色像素点的位置
    left = 0
    for col in range(thresh.shape[1]):
        if thresh[:, col].sum() > 0:
            left = col
            break
    # 检测二值图最右边的白色像素点的位置
    right = 0
    for col in range(thresh.shape[1] - 1, -1, -1):
        if thresh[:, col].sum() > 0:
            right = col
            break

    # 检测二值图最上边的白色像素点的位置
    top = 0
    for row in range(thresh.shape[0]):
        if thresh[row, :].sum() > 0:
            top = row
            break

    # 检测二值图最下边的白色像素点的位置
    bottom = 0
    for row in range(thresh.shape[0] - 1, -1, -1):
        if thresh[row, :].sum() > 0:
            bottom = row
            break

    # crop image
    img_cropped = Image.open(img_path)
    img_cropped = img_cropped.crop((left, top, right + 1, bottom + 1))
    mask_imgs = [
        mask_img.crop((left, top, right + 1, bottom + 1)) for mask_img in mask_imgs
    ]

    # padding image
    img_cropped = pad_to_square(img_cropped)
    mask_imgs = [pad_to_square(mask_img) for mask_img in mask_imgs]

    return img_cropped, mask_imgs


def process_single_image(fundus_path):
    file_name = os.path.basename(fundus_path).split(".")[0]
    mask_paths_sameid = MASK_PATH(file_name)

    mask_imgs = [Image.open(mask_path) for mask_path in mask_paths_sameid]
    fundus_img, mask_imgs = border_crop_square_padding(fundus_path, mask_imgs)

    fundus_img = fundus_img.resize((IMG_SIZE, IMG_SIZE))
    mask_imgs = [mask_img.resize((IMG_SIZE, IMG_SIZE)) for mask_img in mask_imgs]

    fundus_dst = f"{SAVE_DIR}/fundus/{file_name}.jpg"
    os.makedirs(os.path.dirname(fundus_dst), exist_ok=True)
    fundus_img.save(fundus_dst)

    for mask_img, mask_path in zip(mask_imgs, mask_paths_sameid):
        # do not save if mask is pure black
        if np.array(mask_img).sum() == 0:
            continue
        lesion_name = mask_path.split("/")[-2]
        mask_dst = f"{SAVE_DIR}/mask/{lesion_name}/{file_name}.jpg"
        os.makedirs(os.path.dirname(mask_dst), exist_ok=True)
        mask_img.save(mask_dst)


if __name__ == "__main__":
    success_cnt, failed_cnt = 0, 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_single_image, fundus_path)
            for fundus_path in FUNDUS_PATH
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()
                success_cnt += 1
            except Exception as e:
                failed_cnt += 1
                print(f"Error processing image: {e}")

    print(f"Successed: {success_cnt}, Failed: {failed_cnt}")
