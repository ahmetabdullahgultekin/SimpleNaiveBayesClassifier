import json

import numpy as np
import pandas as pd

# Constant for the target column
TARGET_COLUMN = 'PlayTennis'


# Load and prepare your dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    print('---------------------------------------------------------\n')
    print('Dataset:')
    print(data)
    print('\n---------------------------------------------------------')
    summarize_data(data)
    return data


# Summarize the dataset
def summarize_data(data):
    print('\nUnique values in each column:\n')
    for column in data.columns:
        print(f'{column}: {data[column].unique()}')
    print('\n---------------------------------------------------------')


# Summarize the occurrences of each value in the target column
def summarize_occurrences(data):
    target_value_counts = data[TARGET_COLUMN]
    # add_data_to_json(data, target_value_counts)
    # print_values_from_json()
    # print_values_from_memory(target_value_counts, data)


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
            # values_file.write((data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json() + ',')
            if column == data.drop(columns=[TARGET_COLUMN]).columns[-1]:
                values_file.write(
                    (data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json())
            else:
                values_file.write(
                    (data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json() + ',')
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
    # print('Priors:')
    # print(priors)
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


# List the classes of the new instance
def predict_the_result(new_instance, target_column):
    # total = P(Outlook = Sunny) * P(Temperature = Cool) * P(Humidity = High) * P(Wind = Strong) * P(PlayTennis = Yes)
    sum_of_yes = 1
    # total = P(Outlook = Sunny) * P(Temperature = Cool) * P(Humidity = High) * P(Wind = Strong) * P(PlayTennis = No)
    sum_of_no = 1
    # Search the classes of the new instance in the json file
    with open('values.json', 'r') as values_file:
        values = json.load(values_file)
        for value in values['Values']:
            for key, instance in new_instance.items():
                # ('Sunny', 'No') is the format in the json file
                filter_string = f"('{instance}', '{target_column}')"
                if filter_string in value:
                    print("---------------------------------------------------------")
                    print(f"key: {key} - value: {instance} - likelihood: {value[filter_string]}")
                    if 0 < value[filter_string] <= 1:
                        if target_column == 'Yes':
                            sum_of_yes *= value[filter_string]
                            print("*******************************************************")
                            print(f"Total1: {sum_of_yes}")
                        else:
                            sum_of_no *= value[filter_string]
                            print("*******************************************************")
                            print(f"Total2: {sum_of_no}")
                    print("---------------------------------------------------------")
                    print(sum_of_yes)
                    print(sum_of_no)


    # Get the values of prediction from the model belongs to the new instance
    # New values array
    new_values = [sum_of_yes, sum_of_no]
    return new_values


# calculate the probabilities of the new instance
def calculate_probabilities(new_instance, data):
    # Multiply the likelihoods of the new instance
    # result = P(Outlook = Sunny | PlayTennis = Yes) * P(Temperature = Cool | PlayTennis = Yes)
    # * P(Humidity = High | PlayTennis = Yes) * P(Wind = Strong | PlayTennis = Yes) * P(PlayTennis = Yes)
    # Iterate over the target column
    for target_value in data[TARGET_COLUMN].unique():
        predict_the_result(new_instance, target_value)
    # result = P(Outlook = Sunny | PlayTennis = No) * P(Temperature = Cool | PlayTennis = No)
    # * P(Humidity = High | PlayTennis = No) * P(Wind = Strong | PlayTennis = No) * P(PlayTennis = No)


# Print the results
def print_results(new_values):
    print("---------------------------------------------------------")
    print(f"Result of the new instance: {new_values}")


# Start the program
if __name__ == "__main__":
    # Enter the new instance
    new_prediction_data = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
    # Load the dataset
    data_set = load_data('play_tennis_dataset.csv')
    # Summarize the dataset
    summarize_occurrences(data_set)
    # Train the Naive Bayes model
    training_model = train_naive_bayes(data_set, TARGET_COLUMN)
    # Get the likelihoods of the new instance
    new_prediction_classes = predict_the_result(new_prediction_data, TARGET_COLUMN)
    # Print the results
    print_results(new_prediction_classes)
    # calculate_probabilities(new_prediction_data, data_set)
