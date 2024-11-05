fdimport pandas as pd
import cv2
from ultralytics import YOLO
import numpy as np
from numpy import array
df = pd.read_csv("cctvStatusD03.csv")
option = ["Hwy 50 at Zinfandel Dr EB 2"]
row_index = 191
url = df.loc[row_index, "streamingVideoURL"]
def draw_rectangle(frame, x, y, w, h, color=(255, 0, 0), thickness = 2):
    cv2.rectangle(frame, (x,y), (x+w, y+h), color, thickness)
video_path = "output.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    draw_rectangle(frame, 0, 13, 1280, 23)
    draw_rectangle(frame, 0, 38, 1280, 26)
    draw_rectangle(frame, 0, 65, 1280, 29)
    draw_rectangle(frame, 0, 95, 1280, 32)
    draw_rectangle(frame, 0, 128, 1280, 36)
    draw_rectangle(frame, 0, 165, 1280, 41)
    draw_rectangle(frame, 0, 297, 1280, 57)
    draw_rectangle(frame, 0, 360, 1280, 65)
    draw_rectangle(frame, 0, 431, 1280, 75)
    draw_rectangle(frame, 0, 512, 1280, 89)
    draw_rectangle(frame, 0, 609, 1280, 100)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# Load the YOLO11 model
model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture(video_path)
car_ids = []
lane1=0
lane2=0
lane3=0
lane4=0
lane5=0
lane6=0
lane7=0
lane8=0
lane9=0
lane10=0
lane11=0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        for result in results:
            boxes = result.boxes

            for box in boxes:
                if box.cls == 2:
                    print(box.xyxy.numpy())
                    if box.id not in car_ids:
                        car_ids.append(box.id)
                        arr=np.array([box.xyxy.numpy()])
                        print(arr[0][0][1])
                        avg = (arr[0][0][1] + arr[0][0][3])/2
                        if (0 <= avg <= 22):
                            lane1=lane1+1
                        elif 22 <= avg <= 42:
                            lane2=lane2+1
                        elif 42<=avg<=71:
                            lane3=lane3+1
                        elif 71<=avg<=103:
                            lane4=lane4+1
                        elif 103<=avg<=142:
                            lane5=lane5+1
                        elif 142<=avg<=183:
                            lane6=lane6+1
                        elif 183<=avg<=314:
                            lane7=lane7+1
                        elif 314<=avg<=387:
                            lane8=lane8+1
                        elif 387<=avg<=452:
                            lane9=lane9+1
                        elif 452<=avg<=539:
                            lane10=lane10+1
                        else:
                            lane11=lane11+1

                        
    

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
print("Lane count top to bottom")
print(lane1)
print(lane2)
print(lane3)
print(lane4)
print(lane5)
print(lane6)
print(lane7)
print(lane8)
print(lane9)
print(lane10)
print(lane11)
