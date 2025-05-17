import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objs as go
import base64
from io import BytesIO
import cv2
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import time

# Define dataset and model paths
base_dir = r"/Users/dhruvloriya/Desktop/PROJECT/archive"
model_path = 'enhanced_stressdetect.keras'

# Load the trained model with error handling
model = None
try:
    if os.path.exists(model_path):
        # Try to load the model with basic options
        model = tf.keras.models.load_model(model_path, compile=False)
        # Recompile the model with a simple configuration
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        st.success("Model loaded successfully!")
    else:
        st.error(f"Model file not found at: {model_path}")
        st.info("Please make sure the model file exists in the correct location.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Creating a simple model as fallback...")
    # Create a simple model as fallback
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        st.warning("Using a simple fallback model. Results may not be accurate.")
    except Exception as e:
        st.error(f"Failed to create fallback model: {str(e)}")
        model = None

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Define a function to make predictions
def predict_stress(image):
    if model is None:
        st.warning("Model not loaded. Using default stress level.")
        return 0.5  # Return a default value if model is not loaded
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        stress_prob = predictions[0][0]  # Probability of stress
        return stress_prob
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return 0.5  # Return default value on prediction error

# Define a function to suggest exercises based on stress percentage
def suggest_exercise(stress_percentage):
    if stress_percentage <= 10:
        exercise = "Maintain a balanced diet and get adequate sleep"
    elif stress_percentage <= 20:
        exercise = "Take a short walk in nature"
    elif stress_percentage <= 30:
        exercise = "Practice mindfulness meditation"
    elif stress_percentage <= 40:
        exercise = "Engage in light physical activity like yoga"
    elif stress_percentage <= 50:
        exercise = "Try breathing exercises"
    elif stress_percentage <= 60:
        exercise = "Listen to calming music"
    elif stress_percentage <= 70:
        exercise = "Talk to a friend or family member"
    elif stress_percentage <= 80:
        exercise = "Take a break and relax"
    elif stress_percentage <= 90:
        exercise = "Engage in a hobby you enjoy"
    else:
        exercise = "Seek professional help if needed"
    return exercise

# Define a function to suggest study techniques based on stress level
def suggest_study_techniques(stress_percentage):
    if stress_percentage <= 20:
        return "Your stress level is low. Focus on challenging topics and try advanced study techniques like the Feynman Technique or spaced repetition."
    elif stress_percentage <= 40:
        return "Your stress level is moderate. Break down complex topics into smaller parts and use the Pomodoro Technique (25 minutes of focused study followed by a 5-minute break)."
    elif stress_percentage <= 60:
        return "Your stress level is elevated. Focus on review and reinforcement of material you already know. Use active recall and practice problems to build confidence."
    elif stress_percentage <= 80:
        return "Your stress level is high. Take frequent breaks and focus on basic concepts. Consider discussing difficult topics with peers or teachers for clarification."
    else:
        return "Your stress level is very high. Consider taking a short break from studying to address your stress. When you return, focus on the most essential material only."

# Define a function to create a 3D scatter plot
def create_3d_plot(stress_percentage):
    frames = []
    for i in range(100):
        x = np.random.randn(100)
        y = np.random.randn(100)
        z = np.random.randn(100) * stress_percentage / 100  # Scale z-axis based on stress level

        frame = go.Frame(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=6,
                color=z,
                colorscale='Viridis',
                colorbar=dict(title='Stress Level'),
                opacity=0.8,
                line=dict(width=1)
            )
        )])
        frames.append(frame)

    layout = go.Layout(
        title=f'3D Visualization of Stress Levels ({stress_percentage:.2f}% Stress)',
        scene=dict(
            xaxis=dict(title='X-axis', showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            yaxis=dict(title='Y-axis', showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            zaxis=dict(title='Z-axis', showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.2, y=1.2, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        width=600,  # Adjust width
        height=400,  # Adjust height
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 100, "redraw": True},
                                       "fromcurrent": True, "mode": "immediate"}])]
        )],
        annotations=[
            dict(
                text=f'Stress Level: {stress_percentage:.2f}%',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0,
                y=1.1,
                font=dict(size=16)
            )
        ]
    )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=layout)
    return fig

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Function to create a stress tracking chart
def create_stress_tracking_chart(stress_history):
    if not stress_history:
        return None
    
    df = pd.DataFrame(stress_history)
    df.columns = ['Date', 'Stress Level']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Stress Level'], marker='o', linestyle='-', linewidth=2, markersize=8)
    ax.set_title('Stress Level Over Time', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stress Level (%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 100)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add a horizontal line at 50% stress level
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax.text(len(df) - 1, 52, 'High Stress Threshold', color='r', fontsize=10)
    
    plt.tight_layout()
    return fig

def record_stress_for_duration(duration=10):
    """Record stress levels for a specified duration."""
    stress_history = []
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open camera. Please check your camera connection.")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Adjust exposure value
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create placeholders
    video_placeholder = st.empty()
    progress_placeholder = st.empty()
    time_placeholder = st.empty()
    
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=duration)
    
    try:
        while datetime.datetime.now() < end_time:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame. Please check your camera.")
                break
            
            # Adjust exposure if frame is too dark
            if np.mean(frame) < 50:
                cap.set(cv2.CAP_PROP_EXPOSURE, -2)
            
            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Process each detected face
            for (x, y, w, h) in faces:
                face = gray_frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48)) / 255.0
                face_resized = np.expand_dims(face_resized, axis=-1)
                face_resized = np.expand_dims(face_resized, axis=0)
                
                # Predict stress level
                stress_level = model.predict(face_resized, verbose=0)[0][0]
                stress_percentage = stress_level * 100
                
                # Record stress level with timestamp
                current_time = datetime.datetime.now()
                elapsed_seconds = (current_time - start_time).total_seconds()
                stress_history.append([elapsed_seconds, stress_percentage])
                
                # Draw rectangle around face and display stress level
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Stress: {stress_percentage:.1f}%", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Convert BGR to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Show progress
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            progress = min(100, (elapsed / duration) * 100)
            progress_placeholder.progress(int(progress))
            time_placeholder.markdown(f"<div class='prediction'><h2>Recording: {elapsed:.1f}s / {duration}s</h2></div>", unsafe_allow_html=True)
            
            # Add a small delay to control frame rate
            time.sleep(0.1)
            
    except Exception as e:
        st.error(f"Error during recording: {e}")
    finally:
        # Always release the camera
        cap.release()
    
    return stress_history

def create_stress_plot(stress_history):
    """Create a detailed stress level plot."""
    if not stress_history:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(stress_history, columns=['Time (s)', 'Stress Level (%)'])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time (s)'], df['Stress Level (%)'], 'b-', linewidth=2, label='Stress Level')
    
    # Add horizontal lines for stress thresholds
    plt.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='High Stress Threshold')
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Low Stress Threshold')
    
    # Calculate and plot moving average
    window_size = 5
    df['Moving Average'] = df['Stress Level (%)'].rolling(window=window_size).mean()
    plt.plot(df['Time (s)'], df['Moving Average'], 'r--', linewidth=2, label=f'{window_size}-point Moving Average')
    
    # Add annotations for min and max points
    min_idx = df['Stress Level (%)'].idxmin()
    max_idx = df['Stress Level (%)'].idxmax()
    plt.annotate(f'Min: {df["Stress Level (%)"].min():.1f}%', 
                xy=(df['Time (s)'][min_idx], df['Stress Level (%)'][min_idx]),
                xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Max: {df["Stress Level (%)"].max():.1f}%', 
                xy=(df['Time (s)'][max_idx], df['Stress Level (%)'][max_idx]),
                xytext=(10, -10), textcoords='offset points')
    
    # Customize the plot
    plt.title('Stress Level During 10-Second Recording', fontsize=14, pad=20)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Stress Level (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.ylim(0, 100)
    
    # Add stress zones
    plt.fill_between(df['Time (s)'], 0, 30, color='green', alpha=0.1, label='Low Stress Zone')
    plt.fill_between(df['Time (s)'], 30, 60, color='yellow', alpha=0.1, label='Moderate Stress Zone')
    plt.fill_between(df['Time (s)'], 60, 100, color='red', alpha=0.1, label='High Stress Zone')
    
    plt.tight_layout()
    return plt.gcf()

def real_time_camera_detection():
    """Perform real-time stress detection using webcam."""
    try:
        # Define educational content
        educational_content = {
            "Low Stress": {
                "title": "Optimal Learning Zone",
                "content": "You're in the optimal learning zone! This is the perfect time to tackle challenging topics and use advanced study techniques.",
                "tips": [
                    "Try the Feynman Technique to deepen your understanding",
                    "Use spaced repetition for long-term retention",
                    "Challenge yourself with complex problems",
                    "Teach concepts to others to reinforce learning"
                ]
            },
            "Moderate Stress": {
                "title": "Balanced Learning Zone",
                "content": "You're in a balanced learning state. Break down complex topics into manageable chunks.",
                "tips": [
                    "Use the Pomodoro Technique (25 minutes study, 5 minutes break)",
                    "Create mind maps to organize information",
                    "Practice active recall with flashcards",
                    "Take regular breaks to maintain focus"
                ]
            },
            "High Stress": {
                "title": "Stress Management Zone",
                "content": "Your stress level is elevated. Focus on stress management before tackling new material.",
                "tips": [
                    "Practice deep breathing exercises",
                    "Take a short walk to clear your mind",
                    "Use positive self-talk to build confidence",
                    "Break study sessions into smaller chunks"
                ]
            }
        }

        # Check if model is loaded
        if model is None:
            st.error("Model not loaded. Please check the model file.")
            return
        
        # Initialize face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create tabs for different camera modes
        tab1, tab2 = st.tabs(["Continuous Monitoring", "10-Second Recording"])
        
        with tab1:
            st.markdown("<div class='subheader'><h2>Continuous Stress Monitoring</h2></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='feature-card'>
                <h3>How This Works</h3>
                <p>This feature continuously monitors your stress levels in real-time. 
                You can stop the monitoring at any time by clicking the stop button.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize session state for camera control
            if 'camera_active' not in st.session_state:
                st.session_state.camera_active = False
            
            # Create a single button for camera control
            if not st.session_state.camera_active:
                if st.button("Start Camera", key="camera_control_btn"):
                    st.session_state.camera_active = True
                    st.rerun()
            else:
                if st.button("Stop Camera", key="camera_control_btn"):
                    st.session_state.camera_active = False
                    st.rerun()
            
            if st.session_state.camera_active:
                # Initialize camera
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Failed to open camera. Please check your camera connection.")
                    st.session_state.camera_active = False
                    return
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual exposure
                cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Adjust exposure value
                
                # Create placeholders
                video_placeholder = st.empty()
                stress_placeholder = st.empty()
                exercise_placeholder = st.empty()
                study_placeholder = st.empty()
                educational_placeholder = st.empty()
                
                try:
                    while st.session_state.camera_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame. Please check your camera.")
                            break
                        
                        # Adjust exposure if frame is too dark
                        if np.mean(frame) < 50:
                            cap.set(cv2.CAP_PROP_EXPOSURE, -2)
                        
                        # Convert frame to grayscale for face detection
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        # Process each detected face
                        for (x, y, w, h) in faces:
                            face = gray_frame[y:y+h, x:x+w]
                            face_resized = cv2.resize(face, (48, 48)) / 255.0
                            face_resized = np.expand_dims(face_resized, axis=-1)
                            face_resized = np.expand_dims(face_resized, axis=0)
                            
                            # Predict stress level
                            stress_level = model.predict(face_resized, verbose=0)[0][0]
                            stress_percentage = stress_level * 100
                            
                            # Determine stress category
                            stress_category = (
                                "High Stress" if stress_level > 0.6 else
                                "Moderate Stress" if stress_level > 0.3 else
                                "Low Stress"
                            )
                            
                            # Draw rectangle around face and display stress level
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(frame, f"{stress_category} ({stress_level:.2f})", 
                                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
                            # Display stress level and suggest exercise
                            stress_placeholder.markdown(f"<div class='prediction'><h2>Detected Stress Level: {stress_percentage:.2f}%</h2></div>", unsafe_allow_html=True)
                            exercise = suggest_exercise(stress_percentage)
                            exercise_placeholder.markdown(f"<div class='subheader'><h2 background-color='lightseagreen'>{exercise}</h2></div>", unsafe_allow_html=True)
                            
                            # Suggest study techniques based on stress level
                            study_technique = suggest_study_techniques(stress_percentage)
                            study_placeholder.markdown(f"<div class='study-technique'><h3>Recommended Learning Approach:</h3><p>{study_technique}</p></div>", unsafe_allow_html=True)
                            
                            # Display educational content based on stress level
                            edu_content = educational_content[stress_category]
                            educational_placeholder.markdown(f"""
                            <div class='educational-content'>
                                <h3>{edu_content['title']}</h3>
                                <p>{edu_content['content']}</p>
                                <h4>Learning Tips:</h4>
                                <ul>
                                    {''.join([f'<li>{tip}</li>' for tip in edu_content['tips']])}
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Convert BGR to RGB for display in Streamlit
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Add a small delay to control frame rate
                        time.sleep(0.1)
                
                except Exception as e:
                    st.error(f"Error during continuous monitoring: {e}")
                finally:
                    # Always release the camera
                    cap.release()
                    st.session_state.camera_active = False
        
        with tab2:
            st.markdown("<div class='subheader'><h2>10-Second Stress Recording</h2></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='feature-card'>
                <h3>How This Works</h3>
                <p>This feature will record your stress levels for exactly 10 seconds. 
                The recording will automatically stop after 10 seconds, and you'll see a detailed analysis of your stress patterns during this period.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start 10-Second Recording", key="start_recording_btn"):
                with st.spinner("Recording in progress..."):
                    stress_history = record_stress_for_duration(10)
                
                if stress_history:
                    # Save stress history to CSV
                    df = pd.DataFrame(stress_history, columns=['Time (s)', 'Stress Level (%)'])
                    csv_file = f'stress_history_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    df.to_csv(csv_file, index=False)
                    st.success(f"Stress history saved to {csv_file}")
                    
                    # Display stress tracking chart
                    st.markdown("<div class='subheader'><h3>Detailed Stress Analysis</h3></div>", unsafe_allow_html=True)
                    stress_plot = create_stress_plot(stress_history)
                    if stress_plot:
                        st.pyplot(stress_plot)
                        
                        # Add academic education content about the graph
                        st.markdown("""
                        <div class='educational-content'>
                            <h3>Understanding Your Stress Graph</h3>
                            <p>This graph provides valuable insights into your stress patterns during learning activities:</p>
                            <h4>Key Academic Insights:</h4>
                            <ul>
                                <li><strong>Stress Zones:</strong> The colored areas represent different stress levels:
                                    <ul>
                                        <li>Green (0-30%): Optimal learning zone where you can focus and absorb information effectively</li>
                                        <li>Yellow (30-60%): Moderate stress zone where learning is still possible but may require breaks</li>
                                        <li>Red (60-100%): High stress zone where learning efficiency decreases significantly</li>
                                    </ul>
                                </li>
                                <li><strong>Moving Average (Red Line):</strong> Shows your overall stress trend, helping identify if you're maintaining a productive learning state</li>
                                <li><strong>Peak Points:</strong> Indicate moments of highest stress, which might correspond to challenging academic tasks</li>
                                <li><strong>Valley Points:</strong> Show moments of lowest stress, potentially during easier tasks or breaks</li>
                            </ul>
                            <h4>Academic Performance Implications:</h4>
                            <ul>
                                <li>Consistent stress levels in the green zone indicate optimal learning conditions</li>
                                <li>Frequent spikes into the red zone may suggest the need for different study techniques</li>
                                <li>The moving average helps track your stress management progress over time</li>
                                <li>Understanding these patterns can help optimize your study schedule and techniques</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Calculate statistics
                    stress_levels = [level for _, level in stress_history]
                    avg_stress = np.mean(stress_levels)
                    max_stress = max(stress_levels)
                    min_stress = min(stress_levels)
                    std_stress = np.std(stress_levels)
                    
                    # Display statistics
                    st.markdown(f"""
                    <div class='analysis-section'>
                        <h3>Recording Analysis</h3>
                        <p>During the 10-second recording:</p>
                        <ul>
                            <li>Average Stress Level: {avg_stress:.1f}%</li>
                            <li>Peak Stress Level: {max_stress:.1f}%</li>
                            <li>Lowest Stress Level: {min_stress:.1f}%</li>
                            <li>Stress Variability: {std_stress:.1f}%</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Provide recommendations based on the recording
                    stress_category = (
                        "High Stress" if avg_stress > 60 else
                        "Moderate Stress" if avg_stress > 30 else
                        "Low Stress"
                    )
                    edu_content = educational_content[stress_category]
                    st.markdown(f"""
                    <div class='educational-content'>
                        <h3>Based on Your Recording: {edu_content['title']}</h3>
                        <p>{edu_content['content']}</p>
                        <h4>Recommended Learning Tips:</h4>
                        <ul>
                            {''.join([f'<li>{tip}</li>' for tip in edu_content['tips']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("No stress data was recorded. Please try again.")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check your camera connection and try again.")

# Main function to run the Streamlit app
def main():
    # Include Bootstrap CSS and additional styling
    st.markdown("""
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <style>
        .main {
            background: white;
            color: #333;
            font-family: cursive;
        }
        .header {
            text-align: center;
            padding: 5px;
            color: #00796b;
            background-color: #004d40;
            border-radius: 100px;
            margin-bottom: 20px;
        }
        .subheader {
            text-align: center;
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
            padding: 5px;
            color: #000000;
            background-color: #FFA07A;
            border-radius: 100px;
            margin-bottom: 20px;
        }
        .submoon {
            text-align: center;
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
            padding: 5px;
            color: white;
            background-color: #FFA07A;
            border-radius: 50px;
            margin-bottom: 10px;
        }                
        .prediction {
            text-align: center;
            font-family: Cambria, Cochin, Georgia, Times, 'Times New Roman', serif;
            font-size: 20px;
            padding: 5px;
            color: #ff7043;
        }
        .exercise {
            text-align: left;
            font-size: 15px;
            padding: 5px;
            color: #004d40;
        }
        .study-technique {
            text-align: left;
            font-size: 15px;
            padding: 5px;
            color: #004d40;
            background-color: #E0F7FA;
            border-radius: 10px;
            margin-top: 10px;
        }
        .uploaded-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 400px;
            height: 400px;
            object-fit: cover;
            border: 4px solid black;
            border-radius: 10px;
            box-shadow: 6px 6px 12px rgba(0, 0, 0, 0.6);
        }
        .chatbot-button {
            margin: 20px auto;
            padding: 10px 20px;
            font-size: 16px;
            color: white;
            background-color: cyan;
            border: none;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            text-decoration: none;
        }
        .chatbot-button:hover {
            background-color: #0056b3;
        }
        .table-container {
            margin-top: 20px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .table {
            width: 100%;
            border-collapse: collapse;
        }
        .table th, .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .table th {
            background-color: #004d40;
            color: white;
        }
        .table td a {
            color: #00796b;
            text-decoration: none;
        }
        .table td a:hover {
            text-decoration: underline;
        }
        .education-banner {
            background-color: #4CAF50;
            color: white;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .sdg-goal {
            background-color: #2196F3;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .feature-card {
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .feature-card h3 {
            color: #004d40;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .feature-card p {
            color: #333333;
            font-size: 16px;
            line-height: 1.5;
        }
        .research-card {
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            border-left: 5px solid #4CAF50;
        }
        .research-title {
            color: #2e7d32;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }
        .research-text {
            color: #1b5e20;
            font-size: 18px;
            line-height: 1.6;
            margin-bottom: 20px;
            text-align: justify;
        }
        .key-findings {
            background-color: #f1f8e9;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        .key-findings-title {
            color: #33691e;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .key-findings-list {
            color: #33691e;
            font-size: 16px;
            line-height: 1.8;
        }
        .key-findings-list li {
            margin-bottom: 10px;
            position: relative;
            padding-left: 25px;
        }
        .key-findings-list li:before {
            content: "✓";
            color: #4CAF50;
            font-weight: bold;
            position: absolute;
            left: 0;
        }
        .educational-content {
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .educational-content h3 {
            color: #2e7d32;
            font-size: 20px;
            margin-bottom: 10px;
        }
        .educational-content h4 {
            color: #1b5e20;
            font-size: 16px;
            margin-top: 15px;
        }
        .educational-content ul {
            color: #1b5e20;
            padding-left: 20px;
        }
        .educational-content li {
            margin-bottom: 8px;
        }
        .analysis-section {
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .analysis-section h3 {
            color: #004d40;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .analysis-section ul {
            color: #333;
            padding-left: 20px;
        }
        .analysis-section li {
            margin-bottom: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Streamlit application
    st.markdown("<div class='header'><h1 class='text-white p-3 display-md-2 display-1g-3'>Student Stress Detection for Quality Education</h1></div>", unsafe_allow_html=True)
    
    # Add SDG Goal 4 banner
    st.markdown("""
    <div class='sdg-goal'>
        <h2>Empowering Students Through Stress Management</h2>
        <p>Our innovative tool helps educators identify and address student stress, creating a more supportive learning environment where every student can thrive.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Image Upload", "Real-time Detection", "Educational Resources"])
    
    # Add chatbot link button
    st.markdown('<a href="https://mediafiles.botpress.cloud/c6f82a1a-b39e-4803-9c95-8d7412c27542/webchat/bot.html" target="_blank" class="chatbot-button">Ask an Education Expert</a>', unsafe_allow_html=True)
    
    with tab1:
        st.markdown("<div class='subheader'><h2>Upload a Student Photo to Detect Stress Levels</h2></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='feature-card'>
            <h3>How This Helps Quality Education</h3>
            <p>By identifying stress levels in students, educators can provide timely support and create a more conducive learning environment. 
            This tool helps ensure that no student is left behind due to stress-related issues.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload image
        uploaded_file = st.file_uploader("Choose a student photo...", type="jpg")

        if uploaded_file is not None:
            # Display uploaded image with specified size and styling
            image = Image.open(uploaded_file)
            img_base64 = image_to_base64(image)
            img_html = f"<img src='data:image/jpeg;base64,{img_base64}' class='uploaded-image' />"
            st.markdown(img_html, unsafe_allow_html=True)
            st.write("")
            st.write("Analyzing student stress level...")
            
            # Make prediction
            stress_prob = predict_stress(image)
            stress_percentage = stress_prob * 100
            
            # Display prediction with improved styling
            st.markdown("<div class='subheader'><h2>Student Stress Level Analysis</h2></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction'><h2>Detected Stress Level: {stress_percentage:.2f}%</h2></div>", unsafe_allow_html=True)

            # Suggest exercise based on stress level
            exercise = suggest_exercise(stress_percentage)
            st.markdown("<div class='exercise'><h4>Recommended Stress Management Activity</h4></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='subheader'><h2 background-color='lightseagreen'>{exercise}</h2></div>", unsafe_allow_html=True)
            
            # Suggest study techniques based on stress level
            study_technique = suggest_study_techniques(stress_percentage)
            st.markdown("<div class='study-technique'><h3>Recommended Learning Approach:</h3><p>{study_technique}</p></div>", unsafe_allow_html=True)
            
            # Display 3D scatter plot
            st.markdown("<div class='subheader'><h4>3D Visualization of Stress Levels:</h4></div>", unsafe_allow_html=True)
            fig = create_3d_plot(stress_percentage)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
                <div class='submoon'>
                    <h2>Explanation of the Graph:</h2>
                    <p class='text-muted'>This 3D scatter plot visualizes the detected stress levels from the student's image. 
                    The Z-axis represents the stress percentage, with data points colored according to the Viridis colorscale to show varying stress intensities.
                    The plot is animated, allowing viewers to see dynamic changes in stress levels over time. 
                    An annotation at the top displays the exact stress percentage, providing clear context.
                    This visualization offers an engaging way to understand and analyze student stress levels through a combination of spatial positioning and color coding.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        real_time_camera_detection()
    
    with tab3:
        st.markdown("<div class='subheader'><h2>Educational Resources for Stress Management</h2></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='feature-card'>
            <h3>How This Helps Quality Education</h3>
            <p>Access to quality educational resources on stress management helps both students and educators develop strategies for creating 
            a more effective learning environment. These resources support the holistic development of students, which is essential for quality education.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display table with sections containing links to videos and articles
        st.markdown("<div class='subheader'><h3>Resources for Managing Academic Stress</h3></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='table-container'>
        <table class="table">
            <thead>
                <tr>
                    <th scope="col">Resource Type</th>
                    <th scope="col">Description</th>
                    <th scope="col">Link</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Article 1</td>
                    <td>Managing Academic Stress</td>
                    <td><a href="https://www.helpguide.org/articles/stress/stress-management.htm" target="_blank">Read Article</a></td>
                </tr>
                <tr>
                    <td>Article 2</td>
                    <td>Stress Management for Students</td>
                    <td><a href="https://www.webmd.com/balance/stress-management/stress-management" target="_blank">Read Article</a></td>
                </tr>
                <tr>
                    <td>Article 3</td>
                    <td>How to Manage and Reduce Academic Stress</td>
                    <td><a href="https://www.mentalhealth.org.uk/explore-mental-health/publications/how-manage-and-reduce-stress" target="_blank">Read Article</a></td>
                </tr>
                <tr>
                    <td>Video 1</td>
                    <td>Study Techniques for Reducing Academic Stress</td>
                    <td><a href="https://youtu.be/grfXR6FAsI8?feature=shared" target="_blank">Watch Video</a></td>
                </tr>
                <tr>
                    <td>Video 2</td>
                    <td>Mindfulness Techniques for Students</td>
                    <td><a href="https://youtu.be/TYWI929nZKg?feature=shared" target="_blank">Watch Video</a></td>
                </tr>
                <tr>
                    <td>Video 3</td>
                    <td>Breathing Exercises for Test Anxiety</td>
                    <td><a href="https://youtu.be/m3-O7gPsQK0?feature=shared" target="_blank">Watch Video</a></td>
                </tr>
                <tr>
                    <td>Resource 1</td>
                    <td>UNESCO Resources on Quality Education</td>
                    <td><a href="https://en.unesco.org/themes/education" target="_blank">Access Resources</a></td>
                </tr>
                <tr>
                    <td>Resource 2</td>
                    <td>SDG 4: Quality Education</td>
                    <td><a href="https://sdgs.un.org/goals/goal4" target="_blank">Learn More</a></td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a section for educational research
        st.markdown("<div class='subheader'><h3>Research on Stress and Academic Performance</h3></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='research-card'>
            <div class='research-title'>How This Helps Quality Education</div>
            <div class='research-text'>
                Access to quality educational resources on stress management helps both students and educators develop strategies for creating 
                a more effective learning environment. These resources support the holistic development of students, which is essential for quality education.
            </div>
            <div class='key-findings'>
                <div class='key-findings-title'>Key Findings</div>
                <ul class='key-findings-list'>
                    <li>High stress levels can significantly impact academic performance and learning outcomes.</li>
                    <li>Students who receive support for stress management show improved academic achievement.</li>
                    <li>Creating a supportive learning environment is essential for quality education.</li>
                    <li>Early intervention for stress can prevent academic difficulties and dropout rates.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()