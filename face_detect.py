import cv2
import datetime
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier (cv2.data.haarcascades + "haarcascade_fullbody.xml")

frame_size = (int(cap.get(3)), int(cap.get(4)))

while True:
    #Face Detect
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        cv2.imwrite(f'test_data/{current_time}.png',frame)
        time.sleep(2) #waits 2 seconds so it doesn't capture too many of one person

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255,0,0), 3)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()