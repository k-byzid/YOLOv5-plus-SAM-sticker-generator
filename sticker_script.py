import argparse
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
matplotlib.use('TkAgg')
import PIL
from PIL import Image

from segment_anything import build_sam, SamPredictor
from yolov5 import YOLOv5


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def get_mask(mask, image):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * image
    mask_image = torch.Tensor(mask_image)
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy()).astype(np.uint8)).convert("RGBA")

    return mask_image_pil


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def load_models(YOLO_model_path, SAM_checkpoint):
    device = torch.device('cpu')
    yolov5 = YOLOv5(YOLO_model_path, device)

    sam = build_sam(checkpoint=SAM_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    return yolov5, sam_predictor


def get_YOLO_output(model, image):
    results = model.predict(image)
    box = []

    for detection in results.pred:
        for i in range(len(detection.detach().cpu().numpy())):
            box.append((detection.detach().cpu().numpy())[i][:4])
            box[i] = np.array(box[i])
    box = np.asarray(box)

    return box


def get_SAM_output(model, image, box):
    model.set_image(image)

    if box.shape[0] > 1:
        boxes = torch.Tensor(box, device=model.device)
        transformed_boxes = model.transform.apply_boxes_torch(boxes, image.shape[:2])
        masks, _, _ = model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    else:
        masks, _, _ = model.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )

    return masks

def sticker_generator(masks, image):
    sticker = []
    for mask in masks:
        sticker.append(get_mask(mask, image))

    return sticker


def save_sticker(sticker, directory, image_path):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_heading = 'sticker_'
    file_sub_heading = os.path.splitext(os.path.basename(image_path))[0]
    file_num = []
    file_ending = '.png'

    for i in range(len(sticker)):
        file_num.append('_{}'.format(i + 1))
        file_name = file_heading + file_sub_heading + file_num[i] + file_ending
        if not os.path.exists(os.path.join(directory, file_name)):
            sticker[i].save(os.path.join(directory, file_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Sticker Generator Demo", add_help=True
    )
    parser.add_argument(
        "--input_image", type=str, default='assets/car,jpg', required=False, help="path to image file"
    )
    
    args = parser.parse_args()

    # cfg
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    yolo_model_path = "yolov5/weights/yolov5s.pt"
    image_path = args.input_image
    output_dir = 'results'
    device = 'cpu'
        
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    image = load_image(image_path)

    yolov5, sam_predictor = load_models(yolo_model_path, sam_checkpoint)

    # YOLOv5 implementation
    box = get_YOLO_output(yolov5, image)

    # SAM implementation
    masks = get_SAM_output(sam_predictor, image, box)

    # Sticker generator
    sticker = sticker_generator(masks, image)

    # Save stickers
    save_sticker(sticker, output_dir, image_path)
    
    print("Stickers have been saved at {}".format(output_dir))