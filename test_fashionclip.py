import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from fashion_attributes import COMMON_ATTRIBUTES, CLASS_SPECIFIC_ATTRIBUTES

print("downloading FashionCLIP...")

model_id = "patrickjohncyh/fashion-clip" 
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model.to(device)

model.eval()

print(f"Model loaded on {device.upper()}!\n")


def analyze_clothing(image_path, yolo_class):
    image = Image.open(image_path).convert("RGB")
    results = {}

    def get_best_attribute(attribute_name, options_list):
        prompts = [f"{option} {yolo_class}" for option in options_list]

        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=1)[0] 
        
        best_index = probs.argmax().item()
        best_option = options_list[best_index]
        confidence = probs[best_index].item() * 100 

        return best_option, confidence

    for attr_name, attr_options in COMMON_ATTRIBUTES.items():
        best_val, conf = get_best_attribute(attr_name, attr_options)
        results[attr_name] = {"value": best_val, "confidence": conf}

    if yolo_class in CLASS_SPECIFIC_ATTRIBUTES:
        specific_attrs = CLASS_SPECIFIC_ATTRIBUTES[yolo_class]
        for attr_name, attr_options in specific_attrs.items():
            best_val, conf = get_best_attribute(attr_name, attr_options)
            results[attr_name] = {"value": best_val, "confidence": conf}

    return results

folder_path = "output_images" 

if not os.path.exists(folder_path) or not os.listdir(folder_path):
    print(f"folder '{folder_path}' empty or doesnt exist.")
else:
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)            
            detected_class = None
            for cls in CLASS_SPECIFIC_ATTRIBUTES.keys():
                if cls.replace(" ", "_") in filename.lower() or cls in filename.lower():
                    detected_class = cls
                    break
            
            if not detected_class:
                print(f"\n⚠️ Passing '{filename}': cant determine YOLO class from the name.")
                continue

            print(f"\n🧥 analyzing: {filename} (Class: {detected_class.upper()})")
            
            attributes = analyze_clothing(img_path, detected_class)
            
            for attr_name, data in attributes.items():
                val = data["value"]
                conf = data["confidence"]
                warning = " (Low confidence!)" if conf < 25 else ""
                print(f"  • {attr_name.capitalize()}: {val} [{conf:.1f}%]{warning}")