from ultralytics import YOLO

#model = YOLO('models/yolo5_best.pt')
model = YOLO('yolov8x.pt')


result = model.track('input_videos/input_video.mp4', save=True, conf=0.2)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)