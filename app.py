from flask import Flask,render_template,request
app = Flask(__name__)


from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import vonage
from geopy.geocoders import Nominatim
from credentials import key,secret,phone_num

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=r'model(1).tflite')
interpreter.allocate_tensors()


# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img):
    # img = img.convert('RGB')

    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_batch)

    # Invoke the interpreter
    interpreter.invoke()

    # Get the predictions
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])

    # Convert the probabilities to a class label
    predicted_class = np.argmax(tflite_model_predictions)
    
    return predicted_class


geolocator = Nominatim(user_agent="accident_detection_app")

# Example: Get location information for a specific address
location = geolocator.geocode('Jawaharlal Nehru University')


loc = f"\n Address: {location.address} \n Latitude: {location.latitude} \n Longitude: {location.longitude}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def predict_func(): 
    c=1
    flag=False
    if 'videoFile' not in request.files:
        return render_template('result.html', message='No video file uploaded.')

    video_file = request.files['videoFile']

    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}  # Add more if needed
    if video_file.filename.split('.')[-1].lower() not in allowed_extensions:
        return render_template('result.html', message='Invalid video file format.')

    # Save the video file to a temporary location
    video_filename = secure_filename(video_file.filename)
    video_path = os.path.join('temp', video_filename)  # 'temp' is a directory where you store temporary files
    video_file.save(video_path)

    # Open the video file using cv2.VideoCapture
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()

        # Break the loop if no more frames are available
        # if frame is None:break

        if (c % 10 == 0):  
            resized_frame = tf.keras.preprocessing.image.smart_resize(frame, (224,224), interpolation='bilinear')
            pred = predict(resized_frame)
            if(pred==0):
                flag=True
                break
        
        c += 1
    cap.release()  
    
    if(flag==True):
        status = 'Accident detected! \n'
        
        client = vonage.Client(key=key, secret=secret)
        sms = vonage.Sms(client)

        responseData = sms.send_message(
            {
                "from": "Vonage APIs",
                "to": phone_num,
                "text": status + loc
            }
        )
        if responseData["messages"][0]["status"] == "0":
            message = "Message sent successfully to all helplines"
        else:
            message = "Message failed with error: {responseData['messages'][0]['error-text']}"
    else : message = 'No Accident occured :)'
         
    return render_template('result.html',message=message,status=status,loc=loc)

if(__name__=='__main__'):
    app.run(debug=True) 
