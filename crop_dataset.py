import argparse
import os.path as osp
import cv2
import os
import mmcv
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="preprocess dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--crop-size", nargs="?", default=256, type=int, help="height/width of the patches")
    parser.add_argument("--step", nargs="?", default=240, type=int, help="step size for cropping patches")
    parser.add_argument("--thresh-size", nargs="?", default=0, type=int, help="threshold size for cropping patches")
    parser.add_argument(
        "--compression-level", nargs="?", default=3, type=int, help="compression level when save png images"
    )

    args = parser.parse_args()
    for key, value in vars(args).items():
        print("%s: %s" % (key, value))
    return vars(args)


def read_img(file_path: str):
    img = Image.open(file_path)
    img = np.array(img)
    print(f"image shape={img.shape}")


def crop_dataset(img_path: str, out_dir: str, opt: argparse.ArgumentParser) -> None:

    crop_size = opt["crop_size"]
    step = opt["step"]
    thresh_size = opt["thresh_size"]
    img_name, _ = osp.splitext(osp.basename(img_path))

    img = mmcv.imread(img_path, flag="unchanged")
    # img = np.array(Image.open(img_path))

    if img.ndim == 2:
        h, w = img.shape[:2]
    else:
        raise ValueError(f"Image ndim should be 2, but got {img.ndim}")

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)

    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            patch = img[x : x + crop_size, y : y + crop_size, ...]
            filename = osp.join(out_dir, f"{img_name}_s{index:03d}.png")
            cv2.imwrite(
                filename,
                patch,
                [cv2.IMWRITE_PNG_COMPRESSION, opt["compression_level"]],
            )
            # Image.fromarray(patch).save(filename)
    print(f"Processed {img_name} successfully!!!")


def crop_munich_img():
    args = parse_args()
    munich_img = r"F:\Dataset\multi_sensor_landcover_classification\images\Munich_s1.tif"
    out_dir = r"F:\Dataset\multi_sensor_landcover_classification\crop"
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    read_img(munich_img)
    # crop patches
    crop_dataset(munich_img, out_dir, args)


if __name__ == "__main__":
    crop_munich_img()
