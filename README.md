# CORTEX Health - AI-Powered Medical Diagnosis App

## *Project Overview:*
CORTEX Health is a mobile application designed to aid in diagnosing various diseases using artificial intelligence. The app utilizes custom machine learning models developed with TensorFlow to analyze medical images, such as X-rays and CT scans, for the presence of brain tumors, liver diseases, and pneumonia. The app communicates with a server through APIs to perform the diagnosis and returns the results to the user. Additionally, the app aims to incorporate QR code recognition to analyze textual medical reports for disease prediction. The app's potential impact extends to medical interns and rural communities, offering assistance in medical education and diagnosis in areas with limited access to healthcare services.

*Key Features:*
1. *Image-based Disease Diagnosis:*
   Users can upload X-ray or CT scan images through the app. The custom machine learning model processes these images, identifies the presence of brain tumors, liver diseases, or pneumonia, and provides an accuracy percentage for the diagnosis.

2. *QR Code-Based Diagnosis:*
   The app can capture and decode QR codes present on medical reports. It extracts the textual data from the information, analyzes it using the machine learning model, and predicts the potential disease based on the report's content.

3. *Server-Client Communication:*
   The mobile app communicates with the ML server via APIs. It sends the medical images or QR code data to the server for analysis and receives the diagnosis results to display to the user.

4. *Accurate Disease Prediction:*
   The machine learning model is trained on a diverse and extensive dataset of medical images and reports. The dataset is obtained by collaborating with healthcare institutions, with government validation to ensure the accuracy of training data.

5. *Educational Tool for Intern Doctors:*
   The app serves as a learning platform for medical interns, providing real-world examples of disease diagnoses and their corresponding images. It aids interns in understanding the relationship between medical conditions and diagnostic outputs.

6. *Healthcare Accessibility for Rural Areas:*
   CORTEX Health aims to bridge the gap in healthcare accessibility for rural areas where specialized medical services might be limited. The app offers a preliminary diagnosis that can guide individuals to seek further medical attention.

7. *Future Business Model:*
   Beyond its initial purpose, the app has potential for future upgrades and expansion. This includes incorporating additional disease classifications, expanding the app's capabilities to other medical imaging techniques, and potentially partnering with healthcare providers for a premium service model.

*Outcome and Impact:*
- *Medical Education:* The app supports medical interns in learning about disease diagnosis through practical examples and AI-powered analysis.
- *Rural Healthcare:* The app offers preliminary diagnosis for individuals in rural areas, aiding them in making informed decisions about seeking medical care.
- *Efficient Diagnosis:* The app streamlines the diagnosis process, potentially reducing the time required for initial assessments.
- *Business Opportunity:* The project can be developed into a sustainable business model, catering to medical institutions and individuals seeking accurate and accessible disease diagnosis.

*Challenges:*
- Ensuring Data Accuracy: Collaborating with healthcare institutions and obtaining accurate and diverse datasets for training the machine learning model.
- Regulatory Compliance: Adhering to medical data privacy regulations and obtaining necessary permissions for data usage.
- Model Robustness: Ensuring the machine learning model performs consistently and accurately across a wide range of medical images and reports.

*Future Enhancements:*
- *Expanded Disease Range:* Adding more disease classifications for a broader scope of medical diagnosis.
- *Enhanced User Experience:* Improving the user interface, app responsiveness, and user guidance.
- *Continuous Learning:* Implementing mechanisms for the model to learn from new data and update its predictions over time.

By implementing the CORTEX Health app with these features and considerations, you can create a powerful tool that assists in disease diagnosis and contributes to medical education and healthcare accessibility.

## Installation Instruction

- Clone the repository: `git clone https://github.com/abrar-nazib/cortex-health`
- Navigate to the repository and create a virtual environment with venv `python -m venv venv`
- Activate the environment
  - For Windows: `.\venv\Scripts\activate`
  - For Linux: `source venv/bin/activate`
- Install the required packages: `pip install -r requirements.txt`
- Run the server `uvicorn app.main:app --reload`
  - It will return the server IP and port in `IP:PORT` format
  - Usually, the IP is 127.0.0.1 and the port is 8000

## API testing

For the API documentation, after running the server, go to the `http://IP:PORT/docs` and create requests accordingly

## Client APP

The link to the client application repository is https://github.com/AvikArefin/cortex-health-app
