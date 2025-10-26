##Fake Job Postings Detector

##Overview
-The Fake Job Postings Detector is a web application that helps users identify whether a job posting is likely real or fraudulent. Users can input job descriptions, get predictions, and provide feedback to continuously improve the model over time.
-This project is built using Python, Streamlit, and Machine Learning, specifically a Logistic Regression model trained on job posting text data.

##URL for streamlit website deployed
- URL : https://jobshield-fakejob-posts-detection.streamlit.app/
##Dataset
-Base Dataset: fake_job_postings.csv containing real and fraudulent job listings.
-Feedback Dataset: user_feedback.csv stores user-provided corrections for model improvement.

Preprocessing:
-All text columns (title, company_profile, description, requirements, benefits) are combined.
-Text is cleaned by removing URLs, punctuation, numbers, and extra whitespace.
-User feedback is incorporated to retrain and enhance the model over time.

##Features:
-Predict if a job posting is Real or Fraudulent.
-Show prediction probabilities for each class.
-Collect user feedback to improve the model.
-Retrain model dynamically using new feedback.
-Display model performance metrics: Accuracy, Precision, Recall, and F1 Score.

##Technologies Used:
-Python – core programming language
-Streamlit – web interface
-Pandas & NumPy – data handling and processing
-scikit-learn – TF-IDF vectorization, Logistic Regression, and model evaluation
-Regex (re) – text cleaning

##How to Run
-Clone or download the repository.
-Ensure you have Python 3.8+ installed.
-Install dependencies:
  pip install -r requirements.txt
-Place the base dataset fake_job_postings.csv in the project folder.
-Run the Streamlit app:
  streamlit run milestone4.py
-Open the URL provided in the terminal to access the web app.

##Usage
-Paste a job posting text in the input area.
-Click Predict to see if the job is likely real or fraudulent.
-Optionally, provide feedback if the prediction was correct or incorrect.
-Click Retrain Now to update the model with new feedback.

##Model Performance
-Displayed under the “View Model Performance” section in the app:
-Accuracy – measures overall correctness
-Precision – proportion of predicted fraudulent postings that are actually fraudulent
-Recall – proportion of actual fraudulent postings detected
-F1 Score – harmonic mean of precision and recall

##Notes / Limitations
-The model works best with English job postings.
-Predictions are only as accurate as the training data.
-Retraining is based on feedback collected, so accuracy improves over time with user input.
-MySQL or external database integration has been removed; all feedback is stored locally in a CSV.

##Future Improvements
-Add multi-language support.
-Integrate with a central database for collaborative feedback.
-Improve prediction using advanced NLP models like BERT.
-Add visual analytics for fraudulent posting trends.

##Author
-Ritika Bhasin
-student at IPU'28, Btech-IT
