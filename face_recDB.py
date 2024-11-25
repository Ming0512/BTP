from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

currentname = "unknown"
current_roll_number = ""
current_duration = 0 

last_update_dates = {}

encodingsP = "encodings.pickle"
cascade = "haarcascade_frontalface_default.xml"
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

distance_threshold = 0.5

cred = credentials.Certificate("/home/research/face_rec_final/attendence-71b0f-firebase-adminsdk-rg7ej-0fe3a7b983.json") 
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://attendence-71b0f-default-rtdb.firebaseio.com/'  
})

attendance_ref = db.reference("Attendance") 

def record_attendance_to_firebase(name, roll_number):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    start_time = now.strftime("%H:%M:%S")

    student_ref = attendance_ref.child(roll_number)

    student_data = student_ref.get()
    if student_data is None:
        student_ref.set({
            "name": name,
            "attendance_count": 1, 
            "last_attended_date": date,
        })
        print(f"Attendance recorded for {name} (Roll No: {roll_number}).")
    else:
        last_attended_date = student_data.get("last_attended_date")
        if last_attended_date == date:
            print(f"Attendance already marked for {name} (Roll No: {roll_number}) today.")
        else:
            student_ref.update({
                "attendance_count": 1,
                "last_attended_date": date
            })
            print(f"Attendance updated for {name} (Roll No: {roll_number}).")
vs = cv2.VideoCapture(0) 

if not vs.isOpened():
    print("[ERROR] Could not open video stream.")
    exit(1)
fps = FPS().start()

last_update_time = time.time()

while True:
    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=800)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    roll_numbers = []

    for encoding in encodings:
        distances = face_recognition.face_distance(data["encodings"], encoding)
        min_distance_index = distances.argmin()
        min_distance = distances[min_distance_index]
        if min_distance < distance_threshold:
            name = data["names"][min_distance_index]
            roll_number = data["roll_numbers"][min_distance_index]
            current_date = datetime.now().strftime("%Y-%m-%d")
            

            if currentname != name or (name not in last_update_dates or last_update_dates[name] != current_date):
                currentname = name
                current_roll_number = roll_number
                print(f"Recognized: {currentname}, Roll No: {current_roll_number}")
                record_attendance_to_firebase(currentname, current_roll_number)
                last_update_dates[name] = current_date
        else:
            name = "Unknown"
            roll_number = "Unknown"

        names.append(name)
        roll_numbers.append(roll_number)

    for (name, (top, right, bottom, left)) in zip(names, boxes):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    fps.update()

fps.stop()

# Cleanup
vs.release()
cv2.destroyAllWindows()
