---
title: Car Design Analysis
emoji: 🐨
sdk: gradio
sdk_version: 3.38.0
app_file: app.py
pinned: false
license: mit
---
![Screenshot from 2023-08-05 15-51-03](https://github.com/Suraj-Bhor/CarEpochClassifier/assets/26003031/eb994067-95a0-4bd6-a909-2536991acc85)

### Check out the application:

<a href="https://huggingface.co/spaces/sbhor/Car_Design_Analysis" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange">
</a>


# CarEpochClassifier: A Temporal and Morphological Automotive Analysis System

CarEpochClassifier is an advanced machine learning-based project that employs state-of-the-art computer vision and deep learning techniques to provide comprehensive analysis and insights about cars from their visual data. This system goes beyond conventional image recognition and classification, instead offering chronological (temporal) classification, morphological classification, a modernity score based on model year, and a typicality score indicating the car's typical appearance for its body type and model year.

## Features
- **Temporal Classification:** Estimation of the car's model year.
- **Morphological Classification:** Identifies the body type of the car (e.g., Hatchback, SUV, MPV, Saloon, Convertible).
- **Modernity Score:** Calculates a modernity score based on the estimated model year.
- **Typicality Score:** Compares the car's appearance to the typical appearance of cars in its estimated year range and body type.

## Frameworks and Libraries
The project relies heavily on the following frameworks and libraries:

- PyTorch
- Detectron2
- Gradio
- torchvision
- sklearn
- cv2
- numpy
- PIL

## How to Run

### 1. Clone the Repository
Clone this GitHub repository to your local machine.

```bash
git clone https://github.com/Suraj-Bhor/CarEpochClassifier
```

### 2.  Install Dependencies
This project requires Python 3.8 or above. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Run the Application
Execute the main Python script to start the Gradio interface.

```bash
python app.py
```
You will be presented with a Gradio interface where you can upload images of cars to be classified.

### Overview of the application flowchart:
![carcnn (2)](https://github.com/Suraj-Bhor/CarEpochClassifier/assets/26003031/742e2d64-1aec-4335-b45a-c745e4d29244)




