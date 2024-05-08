from flask import Flask, render_template, Response
import cv2
import os
import tensorflow as tf
import numpy as np

model_name = 'unet-model.keras'
model_path = os.path.join('models', model_name)

model = tf.keras.models.load_model(model_path)

camera = cv2.VideoCapture(0)

app = Flask(__name__)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (256, 256))
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+ frame + 
              b'\r\n\r\n')
    
        
def generate_foreground():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (256, 256))
            x = frame/255.0 
            x = np.expand_dims(x, axis=0) 
            
            y_pred = model.predict(x, verbose=0)[0]     
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred >= 0.5
            y_pred = y_pred.astype(np.int32)
            
            predicted_image = np.expand_dims(y_pred, axis=-1)
            predicted_image = np.concatenate([predicted_image, predicted_image, predicted_image], axis=-1)
            predicted_image = predicted_image * 255
            
            
            _, buffer = cv2.imencode(".jpg", predicted_image)
            frame = buffer.tobytes()
        
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+ frame + 
              b'\r\n\r\n')
        

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_one")
def show_one():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_two")
def show_two():
    return Response(generate_foreground(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
    
    