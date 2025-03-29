
## Fake News Detection System

This project is a comprehensive system designed to detect fake news across various content types, including text articles and images. 
The application utilizes advanced machine learning models, including DistilBERT, BERT, RoBERTa, XLNet, and ALBERT for text classification, along with a Convolutional Neural Network (CNN) for image classification.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Features

- Classify text articles, URLs, and images as real or fake.
- Use multiple NLP models for enhanced classification accuracy.
- Upload custom classifiers and datasets for personalization.
- Visualize model evaluation metrics such as confusion matrices and ROC curves.
- User-friendly mobile application developed with React Native.

## Technologies Used

- **Backend**: Flask, Python, Scikit-learn, PyTorch
- **Frontend**: React Native, Expo
- **Data Handling**: Pandas, NumPy
- **APIs**: Axios for HTTP requests
- **Visualization**: Matplotlib, Seaborn

## Installation

1. Clone the repository:

   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
2. Install the backend requirements: bash
    pip install -r requirements.txt
3. Install frontend dependencies:
  bash
  cd frontend
  npm install
4. Start the Flask backend:
  bash
  python app.py
5. Run the React Native application:
  bash
  npm start

Usage
Text Classification:

Input text or a URL into the appropriate field and click "Classify" to receive results on whether the content is real or fake.
Image Classification:

Upload images directly for classification.

Text Classification :
Customize the application by accessing Text Classification and Select various transformer models or train your own NLP Model.

Trained Models for Image Classification can be found -  https://drive.google.com/drive/folders/1-r9FPPwiAl3_Amo0vvCslaUTQLb9gcRA?usp=sharing


Contributing
Contributions are welcome! If you would like to contribute to this project:

1. Fork the repository.
2. Create a new branch (e.g., feature-branch).
3. Make your changes and commit them.
4. Push to your forked repository.
5. Submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.


### Tips for Personalization:
- Replace "yourusername" in the clone command with your actual GitHub username.
- You may want to elaborate on specific functionalities or provide additional instructions based on your project’s workflow.
- Include screenshots or GIFs in the `README.md` to visually demonstrate the application if applicable.

Feel free to modify any sections to better suit your project’s specifics! Let me know if you need any more help!



