😷 Face Mask Detection using CNN

A Convolutional Neural Network (CNN) built with TensorFlow and Keras to automatically detect whether a person in an image is wearing a mask or not wearing a mask.
This project uses deep learning techniques for binary image classification and can be extended for real-time mask detection using a webcam or CCTV feed.

📁 Project Structure
├── data/
│   ├── with_mask/
│   └── without_mask/
├── models/
│   └── mask_detector.h5
├── src/
│   ├── train_model.py
│   └── predict_image.py
├── README.md
└── requirements.txt

🧠 Model Architecture

A custom CNN designed for binary image classification.

Layers Overview:

Conv2D (32 filters, 3×3 kernel) + ReLU + MaxPooling

Conv2D (64 filters, 3×3 kernel) + ReLU + MaxPooling

Flatten Layer

Dense (128 neurons) + ReLU + Dropout (0.5)

Dense (64 neurons) + ReLU + Dropout (0.5)

Output: Dense (1 neuron, Sigmoid activation)

Optimizer: Adam
Loss Function: Binary Crossentropy
Metric: Accuracy

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/<your-username>/face-mask-detection.git
cd face-mask-detection

2. Create Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # for Linux/Mac
venv\Scripts\activate         # for Windows

3. Install Dependencies
pip install -r requirements.txt

📊 Dataset

You can use:

Your own custom dataset (organized as data/with_mask/ and data/without_mask/)

Or a public dataset such as:

Kaggle Face Mask Detection Dataset

Prajnasb/datasets

Each image is resized to 128×128×3 before training.

🚀 Training the Model

Run the training script:

python src/train_model.py


The model will:

Load and preprocess data (scaling to [0,1])

Split into training & validation sets

Train for specified epochs (default: 15)

Save the trained model to models/mask_detector.h5

🔍 Making Predictions

Run the prediction script to classify a single image:

python src/predict_image.py


When prompted:

Path of the image to be predicted: path/to/image.jpg


Example Output:

The person in the image is wearing a mask 😷


or

The person in the image is not wearing a mask 🙅‍♂️

📈 Results
Metric	Training	Validation
Accuracy	~95%	~92%
Loss	↓ decreasing	stable

(Adjust depending on your actual results.)

To generate a classification report:

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_val, y_pred))

🧩 Key Learnings

Consistent preprocessing (BGR → RGB, scaling) matters a lot.

Output layer must match loss function:

Sigmoid + BinaryCrossentropy → Binary classification

Softmax + SparseCategoricalCrossentropy → Multi-class

Dropout helps reduce overfitting.

Class imbalance can be mitigated via class weights or augmentation.

💡 Future Improvements

Add real-time mask detection using OpenCV and webcam.

Implement data augmentation (rotation, flips, brightness).

Convert model to TensorFlow Lite for mobile deployment.

Deploy on Raspberry Pi for edge AI applications.

🧰 Tech Stack

Python 3.12

TensorFlow / Keras

OpenCV

NumPy, Matplotlib

Scikit-learn

🖋️ Author

Yashaswi Srivastava
👩‍💻 Data Scientist & Developer | Passionate about AI, ML, and Automation
🔗 LinkedIn
 | GitHub

🪪 License

This project is licensed under the MIT License
