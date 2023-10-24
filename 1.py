from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data=r"D:\Helmet-Detection-using-YOLOv8-main\Bike Helmet Detection.v2-more-preprocessing-augmentation.yolov8\data.yaml", epochs=100)  # train the model