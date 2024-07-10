from ultralytics import YOLO


#model = YOLO('yolov8n.yaml')
#results = model.train(data='datasets/dataset/data.yaml', epochs=100, imgsz=640)
# Load your trained classification model
model = YOLO('runs\detect/train8\weights/best.pt')

# Run validation to get metrics - ensure 'data.yaml' is configured correctly
results = model.val(data='datasets\dataset\data.yaml')
print(results.box)