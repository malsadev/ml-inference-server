from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from torchvision.io.image import ImageReadMode, decode_image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks


FOOD_CATEGORIES = {52, 53, 54, 55, 56, 57, 58, 59, 60, 61}
SPOON_CATEGORY = 50
# Midpoint of the 18-35 cm^2 top-down area of a standard spoon.
SPOON_AREA_CM2 = 26.5
app = FastAPI()

INGREDIENT_DENSITY = {
    "banana": 0.91,
    "apple": 0.60,
    "orange": 0.84,
    "broccoli": 0.45,
    "carrot": 1.01,
    "sandwich": 0.75,
    "hot dog": 0.85,
    "pizza": 0.70,
    "donut": 0.65,
    "cake": 0.50,
}

AREA_TO_VOLUME_FACTOR = {
    "apple": 4.5,
    "orange": 4.2,
    "banana": 1.8,
    "carrot": 1.7,
    "broccoli": 2.5,
    "sandwich": 1.2,
    "hot dog": 1.5,
    "pizza": 1.0,
    "donut": 1.8,
    "cake": 1.3,
}

# Nutrients per 100 g of edible portion. Approximate values based on USDA
# FoodData Central; prepared foods (sandwich, pizza, ...) vary a lot by recipe,
# so a representative generic version is used for each.
NUTRIENTS_PER_100G = {
    #             kcal, protein_g, fat_g, carbs_g, fiber_g, sugar_g
    "banana":   {"calories_kcal": 89,  "protein_g": 1.1,  "fat_g": 0.3,  "carbs_g": 22.8, "fiber_g": 2.6, "sugar_g": 12.2},
    "apple":    {"calories_kcal": 52,  "protein_g": 0.3,  "fat_g": 0.2,  "carbs_g": 13.8, "fiber_g": 2.4, "sugar_g": 10.4},
    "orange":   {"calories_kcal": 47,  "protein_g": 0.9,  "fat_g": 0.1,  "carbs_g": 11.8, "fiber_g": 2.4, "sugar_g": 9.4},
    "broccoli": {"calories_kcal": 34,  "protein_g": 2.8,  "fat_g": 0.4,  "carbs_g": 6.6,  "fiber_g": 2.6, "sugar_g": 1.7},
    "carrot":   {"calories_kcal": 41,  "protein_g": 0.9,  "fat_g": 0.2,  "carbs_g": 9.6,  "fiber_g": 2.8, "sugar_g": 4.7},
    "sandwich": {"calories_kcal": 250, "protein_g": 12.0, "fat_g": 10.0, "carbs_g": 28.0, "fiber_g": 2.0, "sugar_g": 4.0},  # ham & cheese
    "hot dog":  {"calories_kcal": 247, "protein_g": 10.6, "fat_g": 14.8, "carbs_g": 18.4, "fiber_g": 0.8, "sugar_g": 3.6},  # with bun
    "pizza":    {"calories_kcal": 266, "protein_g": 11.4, "fat_g": 10.4, "carbs_g": 33.0, "fiber_g": 2.3, "sugar_g": 3.6},  # cheese
    "donut":    {"calories_kcal": 421, "protein_g": 5.0,  "fat_g": 22.9, "carbs_g": 49.4, "fiber_g": 1.7, "sugar_g": 22.0},  # plain cake donut
    "cake":     {"calories_kcal": 371, "protein_g": 4.1,  "fat_g": 16.4, "carbs_g": 54.6, "fiber_g": 1.8, "sugar_g": 36.7},  # frosted chocolate
}

@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
    global model, preprocess, categories
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    categories = weights.meta["categories"]
    model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.eval()
    preprocess = weights.transforms()


@app.get("/categories")
def read_categories():
    return categories

@app.post("/analyze")
async def analyze(image: UploadFile):
    # Run model on image
    data = await image.read()
    try:
        img = decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB)
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded file as an image.")
    with torch.no_grad():
        prediction = model([preprocess(img)])[0]
    labels = prediction["labels"].tolist()
    # Masks are per-pixel probabilities of shape (N, 1, H, W); threshold them.
    mask_px = (prediction["masks"].squeeze(1) > 0.5).sum(dim=(1, 2)).tolist()

    # Get mask of spoon (i.e., number of pixels used in the mask) and calculate pixel-to-cm^2 ratio. If no spoon is detected, image can't analyzed.
    # Detections are sorted by score, so the first spoon is the most confident one.
    spoon_px = next((px for label, px in zip(labels, mask_px) if label == SPOON_CATEGORY), None)
    if not spoon_px:
        raise HTTPException(status_code=422, detail="No spoon detected; a spoon is needed as a size reference.")
    px_cm2_ratio = spoon_px / SPOON_AREA_CM2

    weights_g = []
    for label, px in zip(labels, mask_px):
        if label not in FOOD_CATEGORIES:
            continue
        ingredient = categories[label]
        # Estimate the area of all non spoon objects
        area_2d = estimate_area_2d(px, px_cm2_ratio)
        # Estimate the volume of all non spoon objects
        volume = estimate_volume(area_2d, AREA_TO_VOLUME_FACTOR[ingredient])
        # Estimate the weight of all non spoon objects
        weights_g.append((ingredient, estimate_weight(volume, INGREDIENT_DENSITY[ingredient])))

    # compute the nutrient breakdown and return a table
    return compute_nutrient_breakdown(weights_g)


def compute_nutrients(ingredient: str, weight_g: float) -> dict[str, float]:
    """Scale the per-100 g nutrient profile of `ingredient` to `weight_g` grams."""
    per_100g = NUTRIENTS_PER_100G[ingredient]
    scale = weight_g / 100
    return {nutrient: round(value * scale, 1) for nutrient, value in per_100g.items()}


def compute_nutrient_breakdown(weights_g: list[tuple[str, float]]) -> dict:
    """Nutrients for each detected (ingredient, weight in g) pair, plus the plate total.

    Ingredients without a nutrient profile are reported under "unknown"
    instead of failing the whole request.
    """
    items = []
    unknown = []
    totals = {nutrient: 0.0 for nutrient in next(iter(NUTRIENTS_PER_100G.values()))}
    for ingredient, weight_g in weights_g:
        if ingredient not in NUTRIENTS_PER_100G:
            unknown.append(ingredient)
            continue
        nutrients = compute_nutrients(ingredient, weight_g)
        items.append({"ingredient": ingredient, "weight_g": round(weight_g, 1), **nutrients})
        for nutrient, value in nutrients.items():
            totals[nutrient] += value
    totals = {nutrient: round(value, 1) for nutrient, value in totals.items()}
    totals["weight_g"] = round(sum(item["weight_g"] for item in items), 1)
    return {"items": items, "total": totals, "unknown": unknown}


def estimate_weight(volume, density):
    # g = cm^3 * g/cm^3
    return volume * density

def estimate_volume(area_2d, area_to_volume_factor): # Multiply by a constant for now
    # cm^3 = cm^2 * cm (the factor acts as an average thickness)
    return area_2d * area_to_volume_factor

def estimate_area_2d(num_px_mask, px_cm2_ratio): # Need pixel to cm ratio, using a reference object (e.g., spoon)
    # The 2D projected area (top-down silhouette) of a standard spoon typically ranges from 18 to 35 cm² (2.8 to 5.4 in²), depending on the type of spoon.
    # px_cm2_ratio is pixels per cm^2, derived from the spoon mask.
    return num_px_mask / px_cm2_ratio
