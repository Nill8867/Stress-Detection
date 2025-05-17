# Student Stress Detection for Quality Education

## Project Summary

This project is an innovative web application designed to help educators and students identify and manage stress levels in educational settings. By leveraging computer vision and machine learning, the application provides real-time stress detection through facial expression analysis, offering personalized recommendations for stress management and study techniques.

## Key Features

- **Real-time Stress Detection**: Uses webcam to analyze facial expressions and detect stress levels in real-time
- **Image Upload Analysis**: Allows users to upload student photos for stress level assessment
- **Personalized Recommendations**: Provides tailored study techniques and stress management activities based on detected stress levels
- **Educational Resources**: Offers comprehensive resources on stress management and academic performance
- **Research-Based Approach**: Incorporates findings from educational research on stress and learning outcomes
- **Visual Analytics**: Includes 3D visualizations and tracking charts to monitor stress levels over time

## Technical Implementation

The application is built using:
- **Frontend**: Streamlit for an interactive web interface
- **Machine Learning**: TensorFlow for facial expression analysis and stress detection
- **Computer Vision**: OpenCV for face detection and image processing
- **Data Visualization**: Plotly and Matplotlib for creating interactive charts and visualizations
- **Styling**: Custom CSS for a modern, responsive design

## How It Works

1. The application captures facial expressions through webcam or uploaded images
2. A pre-trained deep learning model analyzes facial features to determine stress levels
3. Based on the detected stress level, the system provides personalized recommendations:
   - Study techniques appropriate for the current stress level
   - Stress management activities to help reduce stress
   - Educational resources relevant to the student's situation
4. The application tracks stress levels over time, providing insights into stress patterns

## Educational Impact

This tool contributes to quality education by:
- Helping educators identify students who may be experiencing high stress
- Providing timely interventions to support stressed students
- Offering evidence-based strategies for managing academic stress
- Creating a more supportive learning environment where all students can thrive
- Empowering students with self-awareness and stress management skills

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/student-stress-detection.git
cd student-stress-detection
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run frontend.py
```

## Dependencies

- TensorFlow
- OpenCV
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Plotly
- PIL (Python Imaging Library)

## Project Structure

- `frontend.py`: Main Streamlit application with UI components
- `new1.py`: Core functionality for stress detection and model training
- `enhanced_stressdetect.keras`: Pre-trained model for stress detection
- `archive/`: Dataset directory containing training and testing images

## Future Enhancements

- Progress tracking for stress management goals
- Integration with learning management systems
- Mobile application for on-the-go stress monitoring
- Expanded educational resources and interactive content
- Personalized dashboards with detailed analytics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

This project aims to create a more supportive educational environment by addressing the critical issue of student stress, ultimately contributing to better academic outcomes and student well-being. 