from mlx_vlm import load, generate
from PIL import Image
import mlx.core as mx

model, processor = load("mlx-community/SmolVLM-500M-Instruct-bf16")

image = Image.open("output_images/item_1_short sleeve top.jpg")

prompt = """
<image>
You are an AI that extracts clothing attributes.
The item category is: short sleeve top.
Ignore the person, background, and accessories.

Return ONLY a valid JSON object. You must guess the values based on the image.
Use this exact format:
{
  "main_color": "insert color here",
  "material": "insert material or unknown",
  "fit": "insert fit or unknown",
  "neckline": "insert neckline type or unknown",
  "sleeves": "insert sleeve length",
  "style": "insert style"
}
"""

output = generate(
    model=model,
    processor=processor,
    image=image,
    prompt=prompt,
    max_tokens=150,       
    temperature=0.0,   
    #repetition_penalty=1.3,
    verbose=True        
)