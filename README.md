# Hand Sign to Text Conversion using AI and Computer Vision

## Overview
This project, **Hand Sign to Text Conversion**, leverages deep learning, computer vision, and real-time gesture recognition to translate hand signs into text. It aims to assist the deaf and hard-of-hearing communities by translating sign language gestures into readable text on-screen, with the potential for further voice conversion.

Using **MediaPipe for landmark detection**, **LSTM neural networks** for sequence prediction, and **OpenCV** for real-time video processing, this project captures, processes, and interprets hand signs dynamically, recognizing core phrases and words foundational in daily communication.

## Motivation
This project seeks to bridge communication gaps by providing a reliable, interactive, and scalable solution for interpreting sign language. Sign language users often face communication barriers when interacting with individuals unfamiliar with their language, and this project aims to address that challenge.

## Features
1. **Real-Time Hand Tracking**: Tracks and identifies hand landmarks and gestures using MediaPipe’s holistic model.
  
2. **Sequential Gesture Recognition**: Uses LSTM neural networks to recognize both static and dynamic gestures.
  
3. **Visual Feedback**: Displays recognized text in real time over the video feed, offering instant feedback.

4. **Customizable Vocabulary**: Currently recognizes a basic vocabulary of phrases and can be expanded by training with additional gestures.

5. **Planned Features**:
   - **Text-to-Speech Integration**: Translating recognized text into spoken words.
   - **Mobile Platform Compatibility**: A mobile app version for broader accessibility.
   - **Expanded Vocabulary**: Additional recognizable signs for greater versatility.

## Technology Stack
- **Python**
- **MediaPipe**: For detecting hand landmarks.
- **OpenCV**: For real-time video capture and processing.
- **NumPy**: For data handling.
- **TensorFlow/Keras**: For training the LSTM model.

## Project Structure
- `main.py`: Main script for running the application.
- `mediapipe_detection.py`: Functions for MediaPipe-based hand detection.
- `draw_landmarks.py`: Functions to overlay landmarks for visualization.
- `keypoints_extraction.py`: Extracts keypoints for model input.
- `model_training.ipynb`: Notebook for training the LSTM model.
- `data`: Directory for sample training and test data.
- `model`: Pre-trained LSTM model for gesture recognition.

## Model Training
The model uses an **LSTM neural network** for sequential gesture recognition:
- **Input Data**: A sequence of keypoints for each hand gesture.
- **Training Process**: Trained on a labeled dataset of gestures corresponding to specific phrases.
- **Performance**: Achieves reliable accuracy for detecting core vocabulary with minimal latency.

## Getting Started

### Prerequisites
- Python 3.7 or later
- Required libraries:
  ```bash
  pip install -r requirements.txt

## Usage Guide

### Basic Gestures
The current model recognizes basic hand signs, such as:
- **Hello**: Sign with an open hand.
- **Thank You**: Gesture with a single hand moving away from the chin.
- **I Love You**: Combination of thumb, index, and pinky fingers extended.

These gestures serve as a starting point and can be customized as needed by adding new gesture data and retraining the model.

### Customizing Gesture Recognition
If you'd like to add or modify gestures:
1. **Collect New Data**: Use `record_data.py` (or similar script) to capture keypoint sequences for each new gesture.
2. **Label the Data**: Organize sequences into labeled folders corresponding to each gesture.
3. **Retrain the Model**: Run the training notebook to integrate new gestures into the model.
4. **Deploy the Model**: Save and load the updated model in `main.py` for real-time detection.

By repeating this process, you can extend the model's vocabulary and customize it for various applications or specific sign languages.

## Troubleshooting

### Common Issues
1. **Model Loading Error**: If you encounter errors loading the model, ensure the model file path (`new_action.h5`) is correct and matches the latest trained model.
2. **Camera Not Working**: Confirm that your webcam is properly connected, and permissions are enabled for Python to access it.
3. **Gesture Not Recognized**: Try moving closer to the camera, ensuring a clear background, and practicing with a steady hand. Adjust lighting for better detection.

### Performance Tips
- **Lighting**: Good lighting significantly improves detection accuracy.
- **Background**: Ensure a simple background to reduce false detections.
- **Camera Position**: Position the camera at an angle that clearly captures your hand without obstructions.

## Frequently Asked Questions (FAQ)

1. **Can I add more gestures?**
   Yes! Collect additional gesture data, label it, and retrain the model as outlined above.

2. **Will this work on a mobile device?**
   The project is currently set up for desktop environments, but it can be adapted for mobile platforms by using TensorFlow Lite or MediaPipe’s mobile libraries.

3. **How accurate is the gesture detection?**
   The model achieves reasonable accuracy for its initial set of gestures. However, accuracy may vary with different lighting, backgrounds, or camera quality. Further training with diverse data will improve performance.

4. **Can I use other sign languages?**
   Yes, you can train the model on hand signs from various sign languages by collecting data for each specific gesture.

## Roadmap
Below is a tentative roadmap for future features and improvements:

- **Phase 1**: Expand basic vocabulary to include more common phrases.
- **Phase 2**: Integrate Text-to-Speech (TTS) for spoken output.
- **Phase 3**: Optimize the model for mobile devices.
- **Phase 4**: Introduce support for different regional sign languages.
- **Phase 5**: Develop a cloud-based interface for collaborative training and customization.

## Community and Contribution
We encourage community contributions to help improve and extend this project! Whether you're a developer, researcher, or someone passionate about accessibility, your input is invaluable. Follow these steps to contribute:
1. **Fork the repository**.
2. **Create a new branch** with your feature or bug fix.
3. **Submit a pull request** and describe the changes you've made.

For significant changes, please open an issue first to discuss your ideas and ensure alignment with project goals.

## Contact and Support
If you have questions, encounter issues, or want to propose new features, feel free to reach out through the repository’s **Issues** tab or contact us directly via email.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software as long as the original license is included in derivative works. See the `LICENSE` file for details.

---

We’re excited to share this project and look forward to seeing how it can be improved and utilized for inclusive communication!
