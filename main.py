import numpy as np
import pandas as pd
import json

from panel.io.resources import json_dumps

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


# Summarize the occurrences of each value in the target column
def summarize_occurrences(data):
    target_value_counts = data[TARGET_COLUMN]
    add_data_to_json(data, target_value_counts)
    # print_values_from_json()
    print_values_from_memory(target_value_counts, data)


# Write the datas to the json file
def add_data_to_json(data, target_value_counts):
    # Open the json file in append mode
    with open('values.json', 'a') as values_file:
        values_file.write('{')
        values_file.write('"Values": [')
        values_file.write(target_value_counts.value_counts().to_json() + ',')
        values_file.write(target_value_counts.value_counts(normalize=True).to_json() + ',')
        for column in data.drop(columns=[TARGET_COLUMN]).columns:
            values_file.write(data.groupby([column, TARGET_COLUMN]).size().to_json() + ',')
            # Do not add comma at the end
            #values_file.write((data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json() + ',')
            if column == data.drop(columns=[TARGET_COLUMN]).columns[-1]:
                values_file.write((data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json())
            else:
                values_file.write((data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json() + ',')
        values_file.write(']')
        values_file.write('}')

# Print the values from the json file
def print_values_from_json():
    with open('values.json', 'r') as values_file:
        values = json.load(values_file)
        print(values)


# Print the values from the memory
def print_values_from_memory(target_value_counts, data):
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
    priors = {}
    priors = data[target_column].value_counts(normalize=True).to_dict()
    print('Priors:')
    print(priors)
    return priors


# Calculate likelihoods
def calculate_likelihoods(data):
    likelihoods_dict = {}
    for column in data.drop(columns=[TARGET_COLUMN]).columns:
        likelihoods_dict[column] = {}
        for value in data[column].unique():
            likelihoods_dict[column][value] = {}
            for target_value in data[TARGET_COLUMN].unique():
                likelihoods_dict[column][value][target_value] = data.groupby(
                    [column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()
    return likelihoods_dict


# Train the Naive Bayes model
def train_naive_bayes(data, target_column):
    priors = calculate_prior(data, target_column)
    likelihoods = calculate_likelihoods(data)
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
    # Get the values of the priors and likelihoods from the model belongs to the instance
    model.get('priors')
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
    summarize_occurrences(data_set)
    calculate_likelihoods(data_set)
    training_model = train_naive_bayes(data_set, TARGET_COLUMN)
    print(training_model)
    new_prediction_data = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
    #prediction = predict(new_prediction_data, training_model)
    # add_data_to_json(training_model)
    # save_model(model, 'naive_bayes_model.json')
    # loaded_model = load_model('naive_bayes_model.json')
    # accuracy = evaluate_model(data, loaded_model, 'Wind')
    # print(f'Accuracy: {accuracy}')
    # Predict a new instance
    # new_instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
    # prediction = predict(new_instance, model)
    # print(f'Prediction for the new instance: {prediction}')
