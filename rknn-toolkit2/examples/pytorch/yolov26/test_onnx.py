from pathlib import Path

import cv2
from ultralytics import YOLO


model = YOLO("./best.onnx")
image_path = "./bus.jpg"
output_path = Path("./bus_result.jpg")

results = model(image_path)
result = results[0]

# plot() 会把框、类别名和置信度直接画到图上。
annotated_image = result.plot(conf=True, labels=True)
cv2.imwrite(str(output_path), annotated_image)

print(f"保存结果图片: {output_path.resolve()}")

for box in result.boxes:
	class_id = int(box.cls.item())
	confidence = float(box.conf.item())
	x1, y1, x2, y2 = box.xyxy[0].tolist()
	class_name = result.names[class_id]
	print(
		f"class={class_name}, conf={confidence:.4f}, "
		f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
	)
