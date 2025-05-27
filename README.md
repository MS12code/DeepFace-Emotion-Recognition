# DeepFace Emotion Recognition

A Python project implemented in **Google Colab** that uses DeepFace and OpenCV to detect faces in images and analyze their emotions. The notebook extracts faces, crops them, and predicts emotions like happy, sad, angry, and neutral with visualizations.

---

## Features

- Detect faces from static images using DeepFace  
- Crop and display detected faces  
- Analyze facial emotions using DeepFace’s pre-trained models  
- Visualize emotion probabilities and dominant emotion  
- Handles face detection errors gracefully  

---

## Technologies Used

- Python 3.x  
- [DeepFace](https://github.com/serengil/deepface)  
- OpenCV  
- Matplotlib  
- Google Colab (cloud notebook environment)  

---

## Use Cases

- **Mental health monitoring:** Detect emotional states in patients through facial analysis  
- **Customer feedback:** Analyze customer emotions during product testing or surveys  
- **Human-computer interaction:** Improve user experience by adapting interfaces based on detected emotions  
- **Security and surveillance:** Recognize suspicious or stressed behavior through facial emotion cues  
- **Educational tools:** Help educators understand student engagement and emotions  

---

## Getting Started

### Prerequisites

- Google account to use [Google Colab](https://colab.research.google.com/) (no local setup required)  
- Internet connection  

### How to Use

1. Open the notebook in Google Colab.  
2. Upload your own image or use the provided sample images.  
3. Run all cells to perform face detection and emotion analysis with visual outputs.  

---

## Sample Code Snippet (from the Colab notebook)

```python
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Load image
img_bgr = cv2.imread("your_image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Extract faces
faces = DeepFace.extract_faces(img_rgb, enforce_detection=False)
cropped_face = faces[0]['face']

# Display cropped face
plt.imshow(cropped_face)
plt.axis('off')
plt.show()

# Analyze emotions
results = DeepFace.analyze(cropped_face, actions=['emotion'], enforce_detection=False)
print("Emotion probabilities:", results[0]['emotion'])
print("Dominant emotion:", results[0]['dominant_emotion'])
