# URL Phishing Detection

This project implements a URL phishing detection system using Convolutional Neural Networks (CNN) to classify URLs as safe or potentially harmful. The backend is built with Python Flask, providing an API to receive URLs and return their safety status.

## Features

- **URL Classification**: Uses CNN to analyze and classify URLs as safe or phishing.
- **RESTful API**: Built with Flask to handle incoming URL requests and respond with safety assessments.
- **Real-time Detection**: Quickly evaluates URLs and provides immediate feedback on safety.
- **User-friendly Interface**: Easy to use with simple API endpoints.

## Technologies Used

- **Python**: The core programming language for implementation.
- **Flask**: Lightweight web framework for creating the API.
- **TensorFlow/Keras**: Libraries used for building and training the CNN model.
- **NumPy**: For numerical computations and data manipulation.
- **Pandas**: For handling datasets during training and evaluation.
- **scikit-learn**: For additional data preprocessing and evaluation metrics.

## How It Works

1. **Data Collection**: Gather datasets of safe and phishing URLs for training the CNN model.
2. **Model Training**: Use the collected datasets to train a Convolutional Neural Network to distinguish between safe and malicious URLs.
3. **API Setup**: Set up a Flask application to expose an API endpoint for URL analysis.
4. **URL Evaluation**: When a URL is sent to the API, the model processes it and returns whether it is safe or potentially harmful.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/url-phishing-detection.git
   cd url-phishing-detection


