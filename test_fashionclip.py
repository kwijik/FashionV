from mlx_vlm import load, generate
from PIL import Image
import mlx.core as mx

model, processor = load("mlx-community/SmolVLM-500M-Instruct-4bit")

image = Image.open("output_images/item_1_short sleeve top.jpg")

prompt = """
<image>
You are extracting product attributes from a clothing image.
Describe only the garment.
Indicate:
- Main сolor and shades 
- Material (cotton, wool, polyester, denim, etc.)
- Fit (slim fit, regular, relaxed, oversized)
- Neckline (V-neck, round neck, crew neck, polo, etc.)
- Sleeves (long, short, 3/4, raglan, etc.)
- Length
- Style (casual, formal, streetwear, sporty, etc.)
- Fit and other details (pockets, buttons, print, etc.)

"""

output = generate(
    model=model,
    processor=processor,
    image=image,
    prompt=prompt,
    max_tokens=500,
    temperature=0.1,
    verbose=True  
)
