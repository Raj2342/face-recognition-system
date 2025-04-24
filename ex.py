import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# ===== CONFIGURATION SECTION =====
# Firebase Configuration
FIREBASE_CREDENTIALS = "rpiversion2-firebase-adminsdk-fbsvc-8138482526.json"
FIREBASE_DB_URL = "https://rpiversion2-default-rtdb.firebaseio.com/"

# Camera Configuration - Updated with your credentials
CAMERA_CONFIG = {
    "source": "rtsp",          # Using RTSP protocol
    "ip": "10.47.58.149",      # Your camera IP
    "port": "8559",            # Your camera port
    "endpoint": "/live.sdp",   # Common RTSP endpoint (may vary by camera)
    "username": "raj",         # Your username
    "password": "123456789",   # Your password
    "camera_name": "C1"        # Your camera name
}

# Path Configuration
IMAGE_ATTENDANCE_PATH = 'ImageAttendance'
# ===== END CONFIGURATION =====

# Initialize Firebase
cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})

def markAttendance(name, camera_name):
    ref = db.reference('Attendance')
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    
    attendance_data = ref.get()
    record_id = f"{name}_{camera_name}_{dtString[:10]}"  # Unique ID per day
    
    if attendance_data and record_id in attendance_data:
        print(f"{name} already marked present at {camera_name}")
        return

    ref.child(record_id).set({
        'name': name,
        'camera': camera_name,
        'timestamp': dtString
    })
    print(f"Attendance marked for {name} at {camera_name}")

def get_camera_stream(config):
    if config["source"] == "http":
        if config["username"] and config["password"]:
            auth = f"{config['username']}:{config['password']}@"
            return f"http://{auth}{config['ip']}:{config['port']}{config['endpoint']}"
        return f"http://{config['ip']}:{config['port']}{config['endpoint']}"
    
    elif config["source"] == "rtsp":
        if config["username"] and config["password"]:
            auth = f"{config['username']}:{config['password']}@"
            return f"rtsp://{auth}{config['ip']}:{config['port']}{config['endpoint']}"
        return f"rtsp://{config['ip']}:{config['port']}{config['endpoint']}"
    
    elif config["source"] == "local":
        return 0  # Default webcam
    
    else:
        raise ValueError("Invalid camera source type")

# Load known faces
images = []
classNames = []
myList = os.listdir(IMAGE_ATTENDANCE_PATH)
print("Images found:", myList)

for cl in myList:
    curImg = cv2.imread(f'{IMAGE_ATTENDANCE_PATH}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

print("Class Names:", classNames)

# Encode known faces
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

# Initialize camera with your specific credentials
stream_url = get_camera_stream(CAMERA_CONFIG)
print(f"Connecting to camera at: {stream_url}")
cap = cv2.VideoCapture(stream_url)

# Special settings for RTSP cameras
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
cap.set(cv2.CAP_PROP_FPS, 15)        # Set reasonable FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))  # H264 codec

while True:
    success, img = cap.read()
    if not success:
        print(f"Failed to capture from {CAMERA_CONFIG['camera_name']}. Retrying...")
        cap.release()
        cap = cv2.VideoCapture(stream_url)
        continue

    # Face recognition processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(f"Detected {name} at {CAMERA_CONFIG['camera_name']}")

                # Scale back face locations
                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                
                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f"{name} ({CAMERA_CONFIG['camera_name']})", 
                           (x1 + 6, y2 - 6), 
                           cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
                markAttendance(name, CAMERA_CONFIG['camera_name'])

    # Display camera name and status
    cv2.putText(img, f"{CAMERA_CONFIG['camera_name']} - Live", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow(CAMERA_CONFIG['camera_name'], img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()