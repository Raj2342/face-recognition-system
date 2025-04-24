import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db


# Initialize Firebase
cred = credentials.Certificate("rpiversion2-firebase-adminsdk-fbsvc-8138482526.json")  # Replace with your Firebase JSON file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rpiversion2-default-rtdb.firebaseio.com/'  # Replace with your Firebase database URL
})     
 
def markAttendance(name):
    ref = db.reference('Attendance')  # Firebase Database Node
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')

    # Check if user already marked present
    attendance_data = ref.get()
    if attendance_data and name in attendance_data:
        print(f"{name} already marked present")
        return

    # Push attendance record to Firebase
    ref.child(name).set({
        'name': name,
        'timestamp': dtString
    })
    print(f" Attendance marked for {name} in Firebase")

    

# Path for storing known images
path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)     
print("Images found:", myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print("Class Names:", classNames) 

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList
 
encodeListKnown = findEncodings(images)
print("Encoding Complete!")


# Open Video Capture
#cap = cv2.VideoCapture("http://192.168.234.180:8080/video")  # Change to your mobile IP Webcam
cap = cv2.VideoCapture(0)
while True: 
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        if len(faceDis) == 0:
            continue

        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print("Detected:", name)

            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
