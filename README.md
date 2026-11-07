# ML Inference Server

A FastAPI server that runs an **instance segmentation model** to detect the
ingredients in a food image, and uses the segmentation results to **estimate
the weight** of each detected ingredient.

## How it works

1. An image is sent to the server.
2. An instance segmentation model (currently
   [Mask R-CNN ResNet-50 FPN v2](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
   from torchvision, pretrained on COCO) detects each ingredient and produces a
   per-pixel mask for it.
3. The masks are used to calculate an estimated weight for every detected
   ingredient.

> **Status:** early development. The model pipeline is exercised in
> `tests/test_general.py`; the API endpoints and the weight calculation are
> being built out.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

PyTorch is installed from the CPU-only wheel index (configured in
`pyproject.toml`), so no GPU is required.

## Getting started

Install the dependencies:

```bash
uv sync
```

Start the development server (auto-reloads on code changes):

```bash
uv run fastapi dev src/ml_inference_server/main.py
```

The server listens on <http://127.0.0.1:8000>. Interactive API docs are
available at <http://127.0.0.1:8000/docs>.

To run it in production mode instead:

```bash
uv run fastapi run src/ml_inference_server/main.py
```

## Running the tests

```bash
uv run pytest
```

The tests run the segmentation model on the images in `tests/assets/` and save
the annotated results to `tests/output/`. The model weights are downloaded on
the first run.

## Credits

Test image credits are listed in [`tests/assets/CREDITS.md`](tests/assets/CREDITS.md).
