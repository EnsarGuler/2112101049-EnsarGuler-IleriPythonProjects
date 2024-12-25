import cv2
from ultralytics import YOLO
import os


model = YOLO("yolov8n.pt")


TRAIN_IMAGES_DIR = 'car/train/images/'
TEST_IMAGES_DIR = 'car/test/images/'


classes_names = ["Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100",
                 "Speed Limit 110", "Speed Limit 120", "Speed Limit 20", "Speed Limit 30",
                 "Speed Limit 40", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
                 "Speed Limit 80", "Speed Limit 90", "Stop"]




def load_images_from_folder(folder):
    loaded_images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            loaded_images.append((img, filename))
    return loaded_images


def detect_and_display(images, window_name):
    for img, filename in images:
        results = model.predict(img)[0]

        if len(results.boxes) == 0:
            print(f"Algılanamadı: {filename}")
        else:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = int(box.cls[0])

                if class_id < len(classes_names):
                    label = f"{classes_names[class_id]} {conf:.2f}"
                else:
                    label = f"Unknown {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, img)
            cv2.waitKey(0)

#burdan sonrası video için
def detect_and_display_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video dosyası açılamadı.")
        return


    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video sonlandı.")
            break

        results = model.predict(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.cls[0])

            if class_id < len(classes_names):
                label = f"{classes_names[class_id]} {conf:.2f}"
            else:
                label = f"Unknown {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        cv2.imshow("Video Detection", frame)

        if output_path:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


print("Model eğitimi başlatılıyor...")
model.train(data='car/data.yaml', epochs=6, imgsz=640)

print("Görsel tahmini başlatılıyor...")
loaded_test_images = load_images_from_folder(TEST_IMAGES_DIR)
detect_and_display(loaded_test_images, "Image Detection")

VIDEO_PATH = 'C:/Users/ensar/PycharmProjects/PythonProjectDeneme/video.mp4'
OUTPUT_VIDEO_PATH = 'output_video.mp4'

print("Video tahmini başlatılıyor...")
detect_and_display_video(VIDEO_PATH, OUTPUT_VIDEO_PATH)