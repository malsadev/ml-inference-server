from pathlib import Path

import pytest
import torch
from torchvision.io.image import decode_image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

ASSETS = Path(__file__).parent / "assets"
OUTPUT = Path(__file__).parent / "output"

# Mask R-CNN is trained on COCO, which includes food classes (apple, orange,
# broccoli, pizza, donut, hot dog, ...). The previous FCN model used the
# Pascal VOC classes, which have no food at all, so its masks were all black.
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
model.eval()
preprocess = weights.transforms()

# Bright, distinct mask colours (the default random colours can come out dark).
PALETTE = ["yellow", "cyan", "magenta", "lime", "orange", "deepskyblue", "red", "white"]


def segment(image_path: Path, output_name: str) -> list[str]:
    img = decode_image(str(image_path))

    with torch.no_grad():
        prediction = model([preprocess(img)])[0]

    labels = [categories[i] for i in prediction["labels"]]
    # Masks are per-pixel probabilities of shape (N, 1, H, W); threshold them.
    masks = prediction["masks"].squeeze(1) > 0.5
    # Draw the largest masks first so big ones (e.g. "dining table") don't
    # paint over the smaller objects sitting on top of them.
    masks = masks[masks.sum(dim=(1, 2)).argsort(descending=True)]

    # draw_segmentation_masks paints pixels where masks overlap black, so draw
    # one mask at a time instead (e.g. food masks overlap the table mask).
    result = img
    for i, mask in enumerate(masks):
        result = draw_segmentation_masks(
            result, mask, alpha=0.5, colors=PALETTE[i % len(PALETTE)]
        )
    OUTPUT.mkdir(exist_ok=True)
    image = to_pil_image(result)
    image.save(OUTPUT / output_name)
    image.show()
    return labels


def test_bananas_and_oranges_segmentation():
    labels = segment(ASSETS / "bananas_and_oranges.jpg", "bananas_and_oranges_segmented.png")
    assert {"banana", "orange"} <= set(labels)


def test_apples_and_carrots_segmentation():
    labels = segment(ASSETS / "apples_and_carrots.jpg", "apples_and_carrots_segmented.png")
    assert {"apple", "carrot"} <= set(labels)
