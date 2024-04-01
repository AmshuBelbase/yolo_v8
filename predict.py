import os
import time
from ultralytics import YOLO
import cv2


# VIDEOS_DIR = os.path.join('.', 'custom_data', 'images',
#                           'val')  # address of video

# VIDEOS_DIR = os.path.join('content', 'gdrive', 'MyDrive',
#                           'DETECTION', 'yolov5', 'videos')  # address of video

# /content/gdrive/MyDrive/DETECTION/yolov5/videos/video10.mp4
# VIDEOS_DIR = os.path.join('.')
# video_path = os.path.join(VIDEOS_DIR, 'video5.mp4')
# video_path_out = '{}_out.mp4'.format(video_path)
cam_source = 0
video_path = cam_source  # for webcam
video_path_out = 'webcam_out.mp4'


cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
H, W, _ = frame.shape

if video_path != cam_source:
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(
        *'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model
count = 0
threshold = 0.75

while ret:
    if (count % 2 == 0):
        print(time.time())
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    if video_path != cam_source:
        out.write(frame)
    else:
        cv2.imshow("output", frame)

    ret, frame = cap.read()
    count += 1
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key > -1:  # Handle both 'q' and any key press
        print("Exiting the program...")
        break


cap.release()
if video_path != 0:
    out.release()
cv2.destroyAllWindows()
