Naive Bayes Classifier for Play Tennis Prediction
Project Overview
This project implements a Naive Bayes classifier to predict whether a game of tennis will be played based on weather conditions. The classifier is built using Python, with data processing performed using the Pandas library and results saved in JSON format.

Objectives
To understand and implement the Naive Bayes algorithm.
To classify instances of weather conditions into "Play Tennis: Yes" or "Play Tennis: No."
To evaluate the performance of the classifier using metrics such as accuracy and the confusion matrix.
Prerequisites
Python 3.x
Pandas
A CSV dataset named play_tennis_dataset.csv
Project Structure
load_data(): Loads and displays the dataset.
summarize_data(): Summarizes unique values in each column.
summarize_occurrences(): Summarizes the occurrences of values in the target column.
train_naive_bayes(): Computes prior probabilities and likelihoods for the Naive Bayes model.
predict_the_result(): Predicts the outcome for a new data instance.
calculate_probabilities(): Computes the probabilities for each class.
print_results(): Prints the prediction results for easy interpretation.
Files
play_tennis_dataset.csv: The input dataset used for the project.
values.json: Stores the computed values (likelihoods and priors) in JSON format.
How to Run
Clone the Repository:

bash
Copy code
git clone <repository-url>
cd <repository-folder>
Install Dependencies: Make sure you have Python and Pandas installed:

bash
Copy code
pip install pandas
Run the Script:

bash
Copy code
python <script_name>.py
Replace <script_name> with the actual name of your Python script file.

How It Works
The script loads the dataset from play_tennis_dataset.csv.
The dataset is summarized, and occurrences of values in the target column are computed.
The Naive Bayes model is trained using prior probabilities and likelihoods.
The model makes predictions based on new weather conditions, and the results are displayed.
Evaluation Metrics
Accuracy: The proportion of correctly classified instances.
Confusion Matrix: Used to evaluate the model's performance.
Example Output
The classifier will output the probabilities and make a prediction for a new set of weather conditions, indicating whether a tennis game is likely to occur.

Limitations
The Naive Bayes algorithm assumes independence between features, which may not hold true in all datasets.
The model’s performance can be affected by the size and quality of the input data.
Author
Ahmet Abdullah Gültekin (150121025)