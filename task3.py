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
import task1
import task2

# Deserialize vectors dictionary
def deserialize_vectors_dictionary(file_name):
    with open(file_name, "r") as read_file:
        vectors_dict = json.load(read_file)
    return vectors_dict

# Deserialize word vector position dicitonary
def deserialize_word_vector_pos_dictionary(file_name):
    with open(file_name, "r") as read_file:
        vectors_dict = json.load(read_file)
    return vectors_dict

# Generate heatmap dataframe using sliding window technique
def generate_heatmap_df(row, vectors_dict, vector_type, word_vector_pos_dict, window_length, shift_length):
    tf_df_input_row = {}
    sensor_id = int(row.sensor_id)
    gesture_id = int(row.gesture_id)
    columns = row.index
    tf_df_input_row['sensor_id'] = sensor_id
    tf_df_input_row['gesture_id'] = gesture_id
    row = row.drop(labels=['sensor_id', 'gesture_id'])
    i=0
    time = 0
    while i < (len(row.index)-window_length):
        if pd.isnull(row[i]):
            break
        k = i
        temp_list = []

        while k < (i + window_length):
            if pd.isnull(row[k]):
                break
            temp_list.append(int(row[k]))
            k += 1

        if len(temp_list) < window_length:
            break;
        temp_list.insert(0, sensor_id)
        tf_df_input_row[str(time)] = vectors_dict[str((gesture_id))][vector_type][word_vector_pos_dict[str(tuple(temp_list))]]
        i += shift_length
        time += 1
    
    row = pd.Series(tf_df_input_row, index = columns) 
    return row

# Read input from user
def load_gesture_id():
    gesture_id = int(input("Enter gesture id: "))
    print("Enter 1 for TF vector heatmap")
    print("Enter 2 for TF-IDF vector heatmap")
    print("Enter 3 for TF-IDF2 vector heatmap")
    vector_type = int(input())
    # gesture_id = 1
    # vector_type = 1
    return gesture_id, vector_type

# Generate heat map
def generate_heat_map():
    gesture_id, vector_type = load_gesture_id()

    print("Deserializing objects from previous tasks...")
    vectors_dictionary_file_name = "Intermediate/vectors.json"
    vectors_dict = deserialize_vectors_dictionary(vectors_dictionary_file_name)
    
    word_vector_pos_dict_file_name = "Intermediate/word_vector_pos_dict.json"
    word_vector_pos_dict = deserialize_word_vector_pos_dictionary(word_vector_pos_dict_file_name)

    data_parameters_file_name = "Intermediate/data_parameters.json"
    data = task2.deserialize_data_parameters(data_parameters_file_name)

    print("Generating heatmap...")
    quantized_df = pd.read_csv('Intermediate/quantized.csv')
    tf_df_heatmap = pd.DataFrame(columns=quantized_df.columns)
    tf_df_heatmap = quantized_df.apply(lambda x: generate_heatmap_df(x, vectors_dict, vector_type-1, word_vector_pos_dict,\
         data['window_length'], data['shift_length']), axis = 1)
    tf_df_heatmap = tf_df_heatmap.set_index(['gesture_id', 'sensor_id'])

    # Generate heatmap from dataframe
    sns.heatmap(tf_df_heatmap.loc[gesture_id, :].dropna(axis=1, how="all"), cmap="Greys")

    # Save heatmap
    plt.savefig(str(gesture_id) + '-' + str(vector_type) + '.png')

    print("Task-3 complete!")

def main():
    while(True):
        print("\n******************** Task-3 **********************")
        print(" Enter 1 to generate heatmap of TF, TF-IDF and TF-IDF2 vectors.")
        print(" Enter 2 to exit")
        option = input("Enter option: ")
        if option == '1':
            task1.create_output_directories()
            generate_heat_map()
        else:
            break

if __name__ == "__main__":
    main()
