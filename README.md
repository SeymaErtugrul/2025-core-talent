# 📷 Camera Movement Detection – Core Talent AI Challenge

This project is a solution for the **ATP Core Talent 2025** challenge.

## ☁️ Live Demo & User Video

🟢 Open the Live Streamlit App -- https://2025-core-talent.streamlit.app  
🟢 User Video of Streamlit App -- https://www.youtube.com/watch?v=jYq_5lJQmFY

---
It is an OpenCV-based application that detects camera and object movements in videos and images.

- **Feature-based methods** (SIFT + FLANN and ORB + BruteMatcher) for camera motion detection
- **Optical flow methods** (Lucas-Kanade & Farneback) for detecting and separating object movement
- A **Streamlit web app** for interactive testing and visualization
- Synthetic and real video test cases to validate detection accuracy

---

## ⚙️ Movement Detection Logic

### Camera Movement Detection:

### How It Works: SIFT & ORB

- Convert frames to **grayscale** to speed up processing and simplify computations  
- Detect keypoints and features using **SIFT** (accurate) or **ORB** (fast)  
- Match features between frames:  
  - **SIFT** uses **FLANN matcher** with ratio test to keep good matches  
  - **ORB** uses **Brute-Force matcher** with Hamming distance for speed  
- Estimate a **homography matrix** to understand overall frame movement  
- Calculate movement values:  
  - **Translation** (shift between frames)  
  - **Determinant** (scale and rotation)  
  - **Identity difference** (how much the frame changes)  
- Combine these into a **movement score** and compare with a threshold  

### Object Motion Detection:

- **Lucas-Kanade**: Tracks a small number of strong points between frames  
- **Farneback**: Computes motion for every pixel (dense optical flow)  

### How It Works: Lucas-Kanade & Farneback Optical Flow

- Both estimate pixel movement between frames (optical flow)  
- **Lucas-Kanade** tracks key points efficiently using pyramids  
- **Farneback** calculates a detailed motion map for all pixels  
- Motion magnitude and direction help detect object movement  
- Movement scores decide if movement is significant  
- Motion vectors and colors show movement visually  

### Object vs Camera Movement Separation

- **Motion density**: Low for objects, high for camera  
- **Angle consistency**: Smooth for camera, scattered for objects  
- **Center ratio**: Camera moves near center, objects off-center  
- **Motion uniformity**: Even for camera, irregular for objects  
- These are new topics for me, so I am still learning and exploring better methods.  

---

## ⚠️ Challenges Faced

- Adjusting Farneback parameters (like winsize, poly_sigma, flow thresholds) to get stable results was difficult.
- Balancing accuracy and speed when choosing between SIFT and ORB.
- Differentiating camera and object motion using rule-based analysis has limitations.
- Encountered various deployment issues like file path problems and markdown rendering errors.
- Still learning the best practices, so I often wondered if there were more efficient or accurate approaches.

---

## 💻 How to Run the App Locally

```bash
git clone https://github.com/SeymaErtugrul/2025-core-talent
cd Desktop\2025-core-talent\2025-core-talent\camera-movement-detection
pip install -r requirements.txt
```

### Streamlit App
```bash
streamlit run app.py
```

### Local Testing
```bash
python lokal_test.py
```

This will test both synthetic and real video inputs and print detection results in the console.

---

## 🖼 Example Input & Output

### Streamlit Input:
![{587F9A12-A19D-45C3-856C-9309B6C60DF7}](https://github.com/user-attachments/assets/adc83f15-596a-4df5-9c8b-e72c13251f19)

### Streamlit Output:
![{AD1B8617-A93A-4BD6-8AD1-7A0C06E88D38}](https://github.com/user-attachments/assets/7d9aee64-3b7f-425a-b2b8-304e0529d212)

### Lokal Test Output (Synthetic Video & Real Video):
![image](https://github.com/user-attachments/assets/4fb27ac4-5133-4439-b4b6-9d86d7dd538e)

---

## 🤔 AI Prompts & Tools Used

- **ChatGPT**: Used for explanation of OpenCV algorithms, code optimization, and README formatting
- **Cursor**: Used as code editor and assistant for refactoring & debugging  
  Also used to compare performance of SIFT/ORB and help resolve Streamlit runtime issues during deployment.

---

## 📁 Project Structure

```bash
camera-movement-detection/
├── app.py               # Streamlit UI
├── movement_detector.py # Camera and object motion logic
├── lokal_test.py        # Automated tests with real & synthetic video
├── TestFolder/          # Sample test videos
├── requirements.txt     # Python dependencies
├── style.css            # CSS styles for the web app
└── README.md            # Project documentation
```

---

## 📄 References

- [OpenCV: Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [OpenCV: Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [Streamlit Docs](https://docs.streamlit.io/)
- [CameraBench Dataset](https://huggingface.co/datasets/camerabench)
