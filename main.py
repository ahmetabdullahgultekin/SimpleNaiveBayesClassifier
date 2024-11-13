import numpy as np
import pandas as pd
import json

# Constant for the target column
TARGET_COLUMN = 'PlayTennis'

# Load and prepare your dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    print('---------------------------------------------------------')
    print(data)
    print('\n---------------------------------------------------------')
    summarize_data(data)
    return data

# Summarize the dataset
def summarize_data(data):
    print('Unique values in each column:')
    for column in data.columns:
        print(f'{column}: {data[column].unique()}')
    print('\n---------------------------------------------------------')
    summarize_occurrences(data)

# Summarize the occurrences of each value in the target column
def summarize_occurrences(data):
    target_value_counts = data[TARGET_COLUMN]
    print('Occurrences of each value in the target column:')
    print(target_value_counts.value_counts())
    print('\n---------------------------------------------------------')
    print('Occurrences of each value in the target column (normalized):')
    print(target_value_counts.value_counts(normalize=True))
    print('\n---------------------------------------------------------')
    # Summarize the occurrences of each value for Yes and No
    for column in data.drop(columns=[TARGET_COLUMN]).columns:
        print(f'Occurrences of each value in the {column} column:')
        print(data.groupby([column, TARGET_COLUMN]).size())
        print('\n---------------------------------------------------------')
        print(f'Occurrences of each value in the {column} column (likelihoods):')
        print(data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size())
        print('\n---------------------------------------------------------')


# Calculate prior probabilities
def calculate_prior(data, target_column):
    priors = data[target_column].value_counts(normalize=True).to_dict()
    print('Priors:')
    print(priors)
    return priors


# Calculate likelihoods with Laplace smoothing
def calculate_likelihoods(data, target_column):
    likelihoods = {}
    for column in data.drop(columns=[target_column]).columns:
        likelihoods[column] = {}
        for value in data[column].unique():
            likelihoods[column][value] = {}
            for target_value in data[target_column].unique():
                count = len(data[(data[column] == value) & (data[target_column] == target_value)])
                total = len(data[data[target_column] == target_value])
                likelihoods[column][value][target_value] = (count + 1) / (total + len(data[column].unique()))
    return likelihoods

# Train the Naive Bayes model
def train_naive_bayes(data, target_column):
    priors = calculate_prior(data, target_column)
    likelihoods = calculate_likelihoods(data, target_column)
    return {'priors': priors, 'likelihoods': likelihoods}

# Save the model as a JSON file
def save_model(model, file_path):
    with open(file_path, 'w') as file:
        json.dump(model, file)

# Load the model from JSON file
def load_model(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Predict class for a single instance
def predict(instance, model):
    priors = model['priors']
    likelihoods = model['likelihoods']
    posteriors = {}
    for class_value in priors.keys():
        posterior = np.log(priors[class_value])
        for feature, value in instance.items():
            if value in likelihoods[feature]:
                posterior += np.log(likelihoods[feature][value].get(class_value, 1e-6))
        posteriors[class_value] = posterior
    return max(posteriors, key=posteriors.get)

# Evaluate the model on test data
def evaluate_model(data, model, target_column):
    correct = 0
    for _, row in data.iterrows():
        instance = row.drop(labels=[target_column]).to_dict()
        if predict(instance, model) == row[target_column]:
            correct += 1
    return correct / len(data)

# Start the program
if __name__ == "__main__":
    data_set = load_data('play_tennis_dataset.csv')
    #label_encoders, data_set = encode_data(data_set)
    calculate_likelihoods(data_set, TARGET_COLUMN)
    training_model = train_naive_bayes(data_set, TARGET_COLUMN)
    #save_model(model, 'naive_bayes_model.json')
    #loaded_model = load_model('naive_bayes_model.json')
    #accuracy = evaluate_model(data, loaded_model, 'Wind')
    #print(f'Accuracy: {accuracy}')