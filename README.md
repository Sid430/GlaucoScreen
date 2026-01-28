GlaucoScreen 👁️

A Deep Learning–Based System for Glaucoma Detection and Progression Monitoring
GlaucoScreen is a full-stack machine learning application that combines retinal image analysis with patient metadata to detect glaucoma and assess disease progression. The system integrates deep learning models, classical machine learning, and a secure web interface to enable low-cost, accessible glaucoma screening.

🚀 System Overview

GlaucoScreen consists of three core components:
Glaucoma Detection (CNN) – Image-based classification
Progression Categorization (CNN) – Stable vs. worsening glaucoma
Risk Refinement (Logistic Regression) – Uses age and gender metadata
These components are orchestrated through a Flask web application that allows users to upload retinal images and receive predictions with confidence scores
app.

🧠 Models Used

Task
Model Type
Input
Glaucoma Detection
CNN (Keras .h5)
Fundus image
Progression Detection
CNN (Keras .h5)
Fundus image
Risk Adjustment
Logistic Regression (.pkl)
Age, Gender

Predictions from image-based models and tabular data are combined using a weighted confidence strategy to improve robustness.

🖥️ Web Application (app.py)

The Flask app provides:
Secure login system
Image upload (PNG/JPG/JPEG)
Patient metadata input (age, gender)
Automated preprocessing
Combined inference pipeline
Human-readable results with confidence scores
Confidence Fusion Logic
Combined Confidence = (3 × Image Confidence + Tabular Confidence) / 4

If glaucoma is detected, a progression analysis is automatically triggered.

📂 Repository Structure

GlaucoScreen/
│
├── app.py                     # Flask web application
├── detection.ipynb            # Glaucoma detection model training
├── segmentation.ipynb         # Optic disc/cup segmentation
├── progression.ipynb          # Disease progression modeling
│
├── glaucoma_detection.h5      # CNN detection model
├── progression_detection.h5   # CNN progression model
├── log_reg_model.pkl          # Logistic regression model
├── scaler.pkl                 # Feature scaler
│
├── uploads/                   # Uploaded images
├── templates/                 # HTML templates
└── static/                    # CSS / assets


▶️ Running the Application

1️⃣ Clone the Repository
git clone https://github.com/Sid430/GlaucoScreen.git
cd GlaucoScreen

2️⃣ Create an Environment

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies

pip install flask flask-session tensorflow numpy pillow joblib scikit-learn

4️⃣ Run the App

python app.py

Then open:
http://localhost:5000

🔐 Demo Login
Username: admin
Password: password


📊 Notebooks
segmentation.ipynb
Optic disc/cup segmentation
ROI enhancement
IOU-based evaluation
detection.ipynb
CNN-based glaucoma classification
Data augmentation & early stopping
AUROC / accuracy evaluation
progression.ipynb
Longitudinal disease modeling
Stable vs. worsening classification

📈 Performance Summary
Task
Metric
Score
Segmentation
IOU
95.11%
Detection
Accuracy
98.68%
Detection
AUC
0.99
Progression
Accuracy
95.88%
Progression
AUC
0.99


🌍 Impact
Reduces dependency on expensive ophthalmic equipment
Enables screening in low-resource settings
Designed for mobile deployment
Demonstrates real-world ML + healthcare integration

🔮 Future Work
Mobile app deployment (iOS / Android)
Multimodal progression modeling (OCT + fundus)
Temporal transformers for longitudinal data
Explainability (Grad-CAM, saliency maps)
Secure cloud-based inference

📜 Citation
If you use this work in research or projects:
@misc{glaucoscreen,
  author = {Milkuri, Siddhartha},
  title = {GlaucoScreen: A Deep Learning-Based System for Glaucoma Detection and Progression Monitoring},
  year = {2026},
  url = {https://github.com/Sid430/GlaucoScreen}
}
