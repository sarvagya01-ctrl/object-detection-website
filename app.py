from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

print("App is starting...")

app = Flask(__name__)
model = YOLO("yolov8n.pt")
custom_labels = {
    "person": "Student",
    "bottle": "Dry Waste",
    "banana": "Wet Waste",
    "apple": "Wet Waste",
    "cup": "Dry Waste",
    "cell phone": "Mobile Detected"
}
# Open webcam
camera = cv2.VideoCapture(0)

def generate_frames():
 
    while True:
        success, frame = camera.read()

        results = model(frame, conf=0.6)
        annotated_frame = results[0].plot()

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)