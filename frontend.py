import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objs as go
import base64
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('enhanced_stressdetect.keras')

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
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    stress_prob = predictions[0][1]  # Probability of stress
    return stress_prob

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
        </style>
    """, unsafe_allow_html=True)

    # Streamlit application
    st.markdown("<div class='header'><h1 class='text-white p-3 display-md-2 display-1g-3'>Stress Detection App</h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'><h2>Upload an image to detect stress levels</h2></div>", unsafe_allow_html=True)
    
    # Add chatbot link button
    st.markdown('<a href="https://mediafiles.botpress.cloud/c6f82a1a-b39e-4803-9c95-8d7412c27542/webchat/bot.html" target="_blank" class="chatbot-button">Open Chatbot</a>', unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display uploaded image with specified size and styling
        image = Image.open(uploaded_file)
        img_base64 = image_to_base64(image)
        img_html = f"<img src='data:image/jpeg;base64,{img_base64}' class='uploaded-image' />"
        st.markdown(img_html, unsafe_allow_html=True)
        st.write("")
        st.write("Classifying...")
        
        # Make prediction
        stress_prob = predict_stress(image)
        stress_percentage = stress_prob * 100
        
        # Display prediction with improved styling
        st.markdown("<div class='subheader'><h2>Stress Level Prediction</h2></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction'><h2>Detected Stress Level: {stress_percentage:.2f}%</h2></div>", unsafe_allow_html=True)

        # Suggest exercise based on stress level
        exercise = suggest_exercise(stress_percentage)
        st.markdown("<div class='exercise'><h4>Suggested Exercise</h4></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subheader'><h2 background-color='lightseagreen'>{exercise}</h2></div>", unsafe_allow_html=True)
        
        # Display 3D scatter plot
        st.markdown("<div class='subheader'><h4>3D Visualization of Stress Levels:</h4></div>", unsafe_allow_html=True)
        fig = create_3d_plot(stress_percentage)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
            <div class='submoon'>
                <h2>Explanation of the Graph:</h2>
                <p class='text-muted'>This 3D scatter plot visualizes the detected stress levels from the uploaded image. 
                The Z-axis represents the stress percentage, with data points colored according to the Viridis colorscale to show varying stress intensities.
                The plot is animated, allowing viewers to see dynamic changes in stress levels over time. 
                An annotation at the top displays the exact stress percentage, providing clear context.
                This visualization offers an engaging way to understand and analyze stress levels through a combination of spatial positioning and color coding.</p>
            </div>
            """, unsafe_allow_html=True)

    # Display table with sections containing links to videos and articles
    st.markdown("<div class='subheader'><h3>Resources for Managing Stress</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='table-container'>
    <table class="table">
        <thead>
            <tr>
                <th scope="col">Section</th>
                <th scope="col">Description</th>
                <th scope="col">Link</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Article 1</td>
                <td>Stress Management</td>
                <td><a href="https://www.helpguide.org/articles/stress/stress-management.htm" target="_blank">Read Article</a></td>
            </tr>
            <tr>
                <td>Article 2</td>
                <td>Stress Management</td>
                <td><a href="https://www.webmd.com/balance/stress-management/stress-management" target="_blank">Read Article</a></td>
            </tr>
            <tr>
                <td>Article 3</td>
                <td>Stress Management</td>
                <td><a href="https://www.mentalhealth.org.uk/explore-mental-health/publications/how-manage-and-reduce-stress" target="_blank">Read Article</a></td>
            </tr>
            <tr>
                <td>Video 1</td>
                <td>Stress Management Techniques</td>
                <td><a href="https://youtu.be/grfXR6FAsI8?feature=shared" target="_blank">Watch Video</a></td>
            </tr>
            <tr>
                <td>Video 2</td>
                <td>Relaxation Techniques for Stress Relief</td>
                <td><a href="https://youtu.be/TYWI929nZKg?feature=shared" target="_blank">Watch Video</a></td>
            </tr>
            <tr>
                <td>Video 3</td>
                <td>Breathing Techniques for Stress</td>
                <td><a href="https://youtu.be/m3-O7gPsQK0?feature=shared" target="_blank">Watch Video</a></td>
            </tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()