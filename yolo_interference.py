from ultralytics import YOLO

model = YOLO('models/last.pt')  # Load model
result = model.predict('input_vedios/demo.mp4' , conf = 0.2 , save = True)
print(result)
print("Boxes")
for box in result[0].boxes:
    print(box)
