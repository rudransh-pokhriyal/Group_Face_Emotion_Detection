import cv2
import numpy as np
from keras.models import model_from_json
import os
import tkinter as tk
from tkinter import Button, Toplevel, filedialog

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the emotion detection model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Function to handle emotion detection loop
def emotion_detection_loop(video_feed_window):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    
    # Check for the 'q' key to quit the emotion detection loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        video_feed_window.destroy()
    else:
        video_feed_window.after(1, lambda: emotion_detection_loop(video_feed_window))

# Function to start emotion detection
def start_emotion_detection():
    global cap
    cap = cv2.VideoCapture(0)
    
    video_feed_window = Toplevel(root)
    video_feed_window.title("Emotion Detection Feed")

    # Create a button to stop emotion detection
    stop_button = Button(video_feed_window, text="Stop Emotion Detection", command=lambda: stop_emotion_detection(video_feed_window))
    stop_button.pack(pady=10)

    # Start the emotion detection loop
    emotion_detection_loop(video_feed_window)

# Function to stop emotion detection
def stop_emotion_detection(video_feed_window):
    cv2.destroyAllWindows()
    cap.release()
    video_feed_window.destroy()

# Function to open a file dialog for image selection
def open_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_local_image(file_path)

# Function to process a local image and display emotions
def process_local_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        roi_gray_frame = gray_img[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(img, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection - Local Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to quit the application
def quit_application():
    root.destroy()

# Create a Tkinter window
root = tk.Tk()
root.title("Emotion Detection")
root.geometry("400x250")  # Set the initial size

# Center the window on the screen
root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = (root.winfo_screenwidth() - width) // 2
y = (root.winfo_screenheight() - height) // 2
root.geometry(f"{width}x{height}+{x}+{y}")

# Create buttons
start_button = Button(root, text="Start Emotion Detection", command=start_emotion_detection)
start_button.pack(pady=10)

image_button = Button(root, text="Local Image", command=open_image)
image_button.pack(pady=10)

quit_button = Button(root, text="Quit", command=quit_application)
quit_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
