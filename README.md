# ML Inference Server

A FastAPI server that runs an **instance segmentation model** to detect the
ingredients in a food image, uses the segmentation results to **estimate the
weight** of each detected ingredient, and returns its **nutrient breakdown**.

## How it works

Send a photo of food; get back each detected food's estimated weight and
nutrients, plus the total.

**The photo must include a spoon.** The spoon is the size reference used to
turn pixels into real-world sizes, so a photo without one is rejected. Shoot
from straight above, with the spoon fully visible and not covered by food.

Only the 10 food classes in the COCO dataset are supported: `banana`, `apple`,
`sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`,
`cake`. Anything else in the photo is ignored.

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

## API

### `POST /analyze`

Send the photo as `multipart/form-data` in a field named `image`:

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -F "image=@tests/assets/generated/02_banana_orange_spoon.jpg"
```

Sample photos that include a spoon are in `tests/assets/generated/`.

**`200 OK`**, with one entry per detected food (two carrots give two `carrot`
entries) and the total for the photo. Weights are in grams, energy in kcal,
and the other nutrients in grams:

```json
{
  "items": [
    {
      "ingredient": "banana",
      "weight_g": 113.3,
      "calories_kcal": 100.8,
      "protein_g": 1.2,
      "fat_g": 0.3,
      "carbs_g": 25.8,
      "fiber_g": 2.9,
      "sugar_g": 13.8
    },
    {
      "ingredient": "orange",
      "weight_g": 163.0,
      "calories_kcal": 76.6,
      "protein_g": 1.5,
      "fat_g": 0.2,
      "carbs_g": 19.2,
      "fiber_g": 3.9,
      "sugar_g": 15.3
    }
  ],
  "total": {
    "calories_kcal": 177.4,
    "protein_g": 2.7,
    "fat_g": 0.5,
    "carbs_g": 45.0,
    "fiber_g": 6.8,
    "sugar_g": 29.1,
    "weight_g": 276.3
  },
  "unknown": []
}
```

**Errors** come back as `{"detail": "..."}`:

| Status | When |
|---|---|
| `400` | The upload isn't a readable image |
| `422` | No spoon was found in the photo |

## Running the tests

```bash
uv run pytest
```

The tests run the segmentation model on the images in `tests/assets/` and save
the annotated results to `tests/output/`. The model weights are downloaded on
the first run.

## Credits

Test image credits are listed in [`tests/assets/CREDITS.md`](tests/assets/CREDITS.md).
