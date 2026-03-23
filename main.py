from ultralytics import YOLO
from PIL import Image
import os

model = YOLO('best.pt')

input_path = 'input_images/test.jpg'
output_dir = 'output_images'

image = Image.open(input_path)

results = model(image)
#print(results)

result = results[0]

print(f"fount items: {len(result.boxes)}")

for i, box in enumerate(result.boxes):
    coords = box.xyxy[0].tolist()
    
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    
    crop_image = image.crop((int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])))
    
    file_name = f"item_{i}_{class_name}.jpg"
    save_path = os.path.join(output_dir, file_name)
    
    crop_image.save(save_path)
    print(f"cropped: {file_name}")

