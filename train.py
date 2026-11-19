
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8n.pt")

    print("Starting YOLOv8 training...")

    results = model.train(
        data=r"C:\Users\priya\YOLOv8_Project\team-alpha-2\data.yaml",
        epochs=50,
        imgsz=416,
        batch=2,
        device="cpu",
        workers=0,
        name="team_alpha_yolov8",
        project="runs/detect",
        patience=10,
        plots=True
    )

    print("Training completed!")
