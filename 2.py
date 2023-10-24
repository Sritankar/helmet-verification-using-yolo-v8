from ultralytics import YOLO
model = YOLO("helmet.pt")  
results = model("helmet.mp4", save=True, show=True)  