import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
import collections
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import quad
import scipy.stats
import numpy as np

# Read and load parameters from user
def load_params():
    data = {}
    data['directory'] = input("Enter directory: ")
    data['window_length'] = int(input("Enter window length: "))
    data['shift_length'] = int(input("Enter shift length: "))
    data['resolution'] = int(input("Enter resolution: "))
    # data['directory'] = "Z"
    # data['window_length'] = 3
    # data['shift_length'] = 2
    # data['resolution'] = 3
    return data

# Load gesture files from the directory
def load_gestures(directory):
    complete_df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"): 
            df = pd.read_csv(directory + "/" + filename, header=None)
            sensor_id = list(range(1, len(df)+1))
            gesture_id = [filename[:-4]] * len(df)
            df['sensor_id'] = sensor_id
            df['gesture_id'] = gesture_id
            complete_df = pd.concat([complete_df, df])
    return complete_df

# Create output directories if it does not exist
def create_output_directories():
    outdir = './Intermediate'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

# Evaluate normal distribution
def normal_distribution_function(x, mean=0, std=0.25):
    value = scipy.stats.norm.pdf(x, mean, std)
    return value

# Get Gaussian band intervals
def get_intervals(r):
    # Normal Distribution
    total_intervals = 2*r
    x_min = -1
    x_max = 1

    interval_spacing = 2/float(2*r)
    x = np.linspace(x_min, x_max, 100)
    
    intervals = []
    count = 0
    start = -1
    while count < 2*r:
        res, err = quad(normal_distribution_function, -1, start)
        intervals.append(res)
        count += 1
        start += interval_spacing
    
    result = {}
    count = 1
    intervals.append(1)
    for i in range(0, len(intervals)):
        intervals[i] = -1 + 2*intervals[i]
    for i in range(0, len(intervals)-1):
        if i == len(intervals)-1:
            result[count] = [intervals[i], 1]
            continue
        result[count] = [intervals[i], intervals[i+1]-0.000000000000000001]
        count += 1
    return result

# Get a dataframe containing minimum and maximum values for each sensor across
# the entire dataset
def get_min_max_df(df):
    min_max_df = pd.DataFrame(columns = ['sensor_id', 'max', 'min'])
    sensor_ids = df.sensor_id.unique()
    for sensor_id in sensor_ids:
        sensor_df = df.loc[df['sensor_id'] == sensor_id]
        just_sensor_value_df = sensor_df.drop(['sensor_id', 'gesture_id'], axis=1)
        max_sensor_value = just_sensor_value_df.max(axis = 0).max()
        min_sensor_value = just_sensor_value_df.min(axis = 0).min()
        min_max_df = min_max_df.append({'sensor_id': str(sensor_id), 'max': max_sensor_value, 'min': min_sensor_value}, ignore_index=True)
    min_max_df = min_max_df.set_index(['sensor_id'])
    return min_max_df

# Normalize each row
def normalize(row, min_max_df):
    sensor_id = row.sensor_id
    max_value = min_max_df.loc[[str(sensor_id)],['max']].values[0][0]
    min_value = min_max_df.loc[[str(sensor_id)],['min']].values[0][0]
    for idx in row.index:
        if idx == 'sensor_id' or idx == 'gesture_id' or pd.isnull(row[idx]):
            continue
        row[idx] = (row[idx] - min_value)/(max_value - min_value)
        row[idx] = row[idx]*2 + -1
    return row

# Quantize each row
def quantize(row, interval_dict):
    sensor_id = row.sensor_id
    for idx in row.index:
        if idx == 'sensor_id' or idx == 'gesture_id' or pd.isnull(row[idx]):
            continue
        for key, value in interval_dict.items():
            if row[idx] >= value[0] and row[idx] <= value[1]:
                row[idx] = int(key)
                break
    return row

# Generate word vectors by using the sliding window technique
def generate_word_vectors(row, word_vector_dict, window_length, shift_length):
    sensor_id = row.sensor_id
    gesture_id = row.gesture_id
    row = row.drop(labels=['sensor_id', 'gesture_id'])
    i=0
    while i < (len(row.index)-window_length):
        if pd.isnull(row[i]):
            break

        temp_key = str((int(gesture_id), sensor_id, i))
        k = i
        temp_list = []

        while k < (i + window_length):
            if pd.isnull(row[k]):
                break
            temp_list.append(int(row[k]))
            k += 1

        if len(temp_list) < window_length:
            break;

        word_vector_dict[temp_key] = tuple(temp_list)
        i += shift_length

    return row

# Delete existing word files
def delete_all_word_files(directory):
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if file.endswith(".wrd")]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)

# Write gesture word files
def write_word_files(directory, word_vector_dict, gesture_ids, sensor_ids):
    delete_all_word_files(directory)
    for key, value in word_vector_dict.items():
        key_tuple = eval(key)
        gesture_id = key_tuple[0]
        sensor_id = key_tuple[1]
        time = key_tuple[2]
        file_path = directory + "/" + str(gesture_id) + ".wrd"
        file1 = open(file_path, "a")  # append mode 
        file1.write(key + " -> " + str(value) + "\n")
        file1.close()

# Serialize gesture word dictionary for future tasks
def serialize_gesture_word_dictionary(word_vector_dict):
    with open("Intermediate/gesture_word_dictionary.json", "w") as write_file:
        json.dump(word_vector_dict, write_file)

# Serialize data paramters for future tasks
def serialize_data_parameters(data):
    with open("Intermediate/data_parameters.json", "w") as write_file:
        json.dump(data, write_file)

# Generates gesture word dictionary
def generate_word_dictionary(data):
    interval_dict = get_intervals(data['resolution'])

    print("Loading gesture files...")
    df = load_gestures(data['directory'])

    print("Normalizing values...")
    min_max_df = get_min_max_df(df)
    min_max_df.to_csv('Intermediate/min_max.csv')

    df = df.apply(lambda x: normalize(x, min_max_df), axis=1)
    df.to_csv('Intermediate/normalized.csv', index=False)

    print("Quantizing values...")
    df = df.apply(lambda x: quantize(x, interval_dict), axis=1)
    df.to_csv('Intermediate/quantized.csv', index=False)
    gesture_ids = df.gesture_id.unique()
    sensor_ids = df.sensor_id.unique()

    print("Generating word files...")
    word_vector_dict = {}
    df.apply(lambda x: generate_word_vectors(x, word_vector_dict, data['window_length'], data['shift_length']), axis = 1)

    print("Writing word files...")
    write_word_files(data['directory'], word_vector_dict, gesture_ids, sensor_ids)

    print("Serializing objects needed for future tasks...")
    serialize_gesture_word_dictionary(word_vector_dict)
    serialize_data_parameters(data)

    print("Task-1 complete!")

def main():
    # Menu
    while(True):
        print("\n******************** Task-1 **********************")
        print(" Enter 1 to generate word dictionary")
        print(" Enter 2 to exit")
        option = input("Enter option: ")
        if option == '1':
            data = load_params()
            create_output_directories()
            generate_word_dictionary(data)
        else:
            break

if __name__ == "__main__":
    main()