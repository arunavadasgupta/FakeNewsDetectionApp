
## Fake News Detection System

This project is a comprehensive system designed to detect fake news across various content types, including text articles and images. 
The application utilizes advanced machine learning models, including DistilBERT, BERT, RoBERTa, XLNet, and ALBERT for text classification, along with a Convolutional Neural Network (CNN) for image classification.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [API](#APIKeys)
- [Screenshots](#Screenshots)

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

1. Clone the repository: git clone https://github.com/arunavadasgupta/fake-news-detection.git cd fake-news-detection
2. Install the backend requirements: bash pip install -r requirements.txt
3. Install frontend dependencies: bash cd frontend npm install
4. Install Expo CLI - https://docs.expo.dev/more/expo-cli/
5. Start the Flask backend: bash python app.py
6. Run the React Native application: bash npm start - You can scan QR or Select approapriate option to run the App in Android or iOS Simulator.


## Usage

Text Classification:

Input text or a URL into the appropriate field and click "Classify" to receive results on whether the content is real or fake.

Image Classification:

Upload images directly for classification.

Text Classification :

Customize the application by accessing Text Classification and Select various transformer models or train your own NLP Model.

Trained Models for Image Classification can be found here - 

https://drive.google.com/drive/folders/1-r9FPPwiAl3_Amo0vvCslaUTQLb9gcRA?usp=sharing



## Contributing

Contributions are welcome! If you would like to contribute to this project:

1. Fork the repository.
2. Create a new branch (e.g., feature-branch).
3. Make your changes and commit them.
4. Push to your forked repository.
5. Submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Demo Video: 
https://vimeo.com/1070678990?share=copy


## API
The homePage.js needs NewsAPI and Google API keys to function properly.
Please update your personal API keys in code before using.

## Screenshots

Click on the Images to view individually on a browser tab.

Home Page :
![image](https://github.com/user-attachments/assets/5a4ae5a6-06fb-4d10-9cf0-42bc60283700)

URL CLassifier : 

![image](https://github.com/user-attachments/assets/5d0f0154-e992-41dd-a233-81c492a92d42)

Text Classification Using Transformer : 

![image](https://github.com/user-attachments/assets/28ab8f91-5b28-4347-ae0f-e3fba9733d23)

Change Transformer Model : 

![image](https://github.com/user-attachments/assets/3fb30e5c-afea-482b-ab5d-86518659f370)


NLP - Training using Default Data : 

![image](https://github.com/user-attachments/assets/b5f5951d-137d-403b-8ea3-a9a2995a7305)

Showing Users Training Results : 

![image](https://github.com/user-attachments/assets/a6565764-d019-4faf-889a-6eddbaeeed58)

Saved Trained Model to Device : 

![image](https://github.com/user-attachments/assets/ab50b3b1-4120-45f3-8de3-cab9f14d1faf)

Classification Using NLP Model : 

![image](https://github.com/user-attachments/assets/937b9606-5473-4b30-8a9e-de7eef7c6155)

Image Classfication : 

![image](https://github.com/user-attachments/assets/ef6f95bc-67a2-4663-8139-38641f9f5aef)

Pick an Impage : 

![image](https://github.com/user-attachments/assets/e1805f6c-8bc5-428d-8077-e65649415c4e)
![image](https://github.com/user-attachments/assets/9b7271fc-5dea-48f7-a431-ca20dfcb0ae2)

Choose Image to Classify :

![image](https://github.com/user-attachments/assets/50e7a886-2af0-4749-9043-95e1a263e924)





