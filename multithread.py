import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import queue
import speech_recognition as sr
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(message):
    engine.say(message)
    engine.runAndWait()

# Load the stress detection model
model_path = 'enhanced_stressdetect.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    print(f"Model file not found at {model_path}. Exiting.")
    exit()

# Camera Thread Class
class CameraThread(threading.Thread):
    def __init__(self, frame_queue, stop_event, resolution=(1280, 720)):
        super().__init__()
        self.cap = cv2.VideoCapture(0)  # Open the default camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.frame_queue = frame_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
        self.cap.release()

# Real-time stress detection
def real_time_camera_detection(frame_queue, stop_event, mode_event):
    """Perform real-time stress detection while listening for voice commands."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = sr.Recognizer()

    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Downscale for faster processing but retain high-resolution display
            small_frame = cv2.resize(frame, (640, 480))
            gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.2,  # Reduce scaling for faster detection
                minNeighbors=5,   # Adjust for fewer false positives
                minSize=(50, 50)  # Minimum face size
            )

            for (x, y, w, h) in faces:
                adjusted_x, adjusted_y, adjusted_w, adjusted_h = x * 2, y * 2, w * 2, h * 2
                adjusted_y = max(adjusted_y - int(0.15 * adjusted_h), 0)  # Shift up slightly

                face = gray_frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (48, 48)) / 255.0
                face_resized = np.expand_dims(face_resized, axis=-1)
                face_resized = np.expand_dims(face_resized, axis=0)

                # Predict stress level
                stress_level = model.predict(face_resized)[0][0]
                stress_category = (
                    "High Stress" if stress_level > 0.6 else
                    "Moderate Stress" if stress_level > 0.3 else
                    "Low Stress"
                )

                # Display stress level
                color = (0, 0, 255) if stress_level > 0.6 else (0, 255, 255) if stress_level > 0.3 else (0, 255, 0)
                label = f"{stress_category} ({stress_level:.2f})"
                cv2.rectangle(frame, (adjusted_x, adjusted_y),
                              (adjusted_x + adjusted_w, adjusted_y + adjusted_h), color, 2)
                cv2.putText(frame, label, (adjusted_x, adjusted_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Display the video feed
            cv2.imshow('Stress Detection', frame)

        # Listen for voice commands
        try:
            with sr.Microphone() as source:
                print("Listening for commands: say 'switch' or 'end'...")
                audio = recognizer.listen(source, timeout=1)
                command = recognizer.recognize_google(audio).lower()
                print(f"Voice command: {command}")

                if "switch" in command:
                    speak("Switching mode.")
                    mode_event.set()
                    return  # Switch to the other mode
                elif "end" in command:
                    speak("Ending detection.")
                    stop_event.set()
                    return
        except sr.UnknownValueError:
            pass  # Ignore unrecognized input
        except Exception as e:
            print(f"Voice recognition error: {e}")

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

# Voice-based stress detection
def voice_stress_detection(stop_event, mode_event):
    recognizer = sr.Recognizer()

    while not stop_event.is_set():
        speak("Please say something for stress analysis or 'switch' to change mode, or 'END' to stop.")
        try:
            with sr.Microphone() as source:
                print("Listening for voice input...")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")

                if "switch" in command:
                    speak("Switching mode.")
                    mode_event.set()
                    return  # Switch to the other mode
                elif "end" in command:
                    speak("Ending detection.")
                    stop_event.set()
                    return
                else:
                    # Simulate stress analysis using voice tone
                    stress_level = np.random.rand()  # Randomly simulate a stress level for now
                    stress_category = (
                        "High Stress" if stress_level > 0.6 else
                        "Moderate Stress" if stress_level > 0.3 else
                        "Low Stress"
                    )
                    speak(f"Detected {stress_category}. Stress level: {stress_level:.2f}.")
        except sr.UnknownValueError:
            print("Speech not understood. Try again.")
        except Exception as e:
            print(f"Error: {e}")

# Main program loop
def main():
    speak("Welcome to stress detection. Say 'camera' for live detection or 'voice' for voice-based detection.")
    recognizer = sr.Recognizer()
    stop_event = threading.Event()

    while not stop_event.is_set():
        mode_event = threading.Event()
        frame_queue = queue.Queue(maxsize=1)
        camera_thread = None  # Initialize camera_thread as None

        try:
            with sr.Microphone() as source:
                print("Listening for mode selection...")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")

                if "camera" in command:
                    speak("Starting camera-based stress detection.")
                    camera_thread = CameraThread(frame_queue, stop_event)
                    camera_thread.start()  # Start the camera thread
                    real_time_camera_detection(frame_queue, stop_event, mode_event)

                elif "voice" in command:
                    speak("Starting voice-based stress detection.")
                    voice_stress_detection(stop_event, mode_event)

                if mode_event.is_set():  # Switch mode if requested
                    continue
                if stop_event.is_set():  # Stop the program if requested
                    break
        except sr.UnknownValueError:
            speak("Sorry, I could not understand. Please say 'camera' or 'voice'.")
            print("Speech not understood. Listening again...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Ensure the thread is joined only if it was started
            if camera_thread and camera_thread.is_alive():
                camera_thread.join()

if __name__ == "__main__":
    main()