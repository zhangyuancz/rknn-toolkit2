from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model
model = YOLO("/home/miles/workspace/rockchip/rknn-toolkit2/rknn-toolkit2/examples/pytorch/yolov26/runs/detect/train/weights/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx", end2end=False, opset=19, half=True)  # export the model to ONNX format
