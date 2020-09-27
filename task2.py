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

# Deserialize gesture word dictionary
def deserialize_gesture_word_dictionary(file_name):
    with open(file_name, "r") as read_file:
        word_vector_dict = json.load(read_file)
    return word_vector_dict

# Intermediate dictionary needed to generate TF vector
def generate_tf_dict(word_vector_dict):
    set_of_all_word_vectors = set()
    tf_dict = {}
    for key, value in word_vector_dict.items():
        key_tuple = eval(key)
        gesture_id = key_tuple[0]
        sensor_id = key_tuple[1]
        temp_key = (gesture_id, sensor_id)
        value.insert(0, sensor_id)
        word = tuple(value)
        
        set_of_all_word_vectors.add(word)
        if temp_key not in tf_dict:
            tf_dict[temp_key] = {}
        internal_dict = tf_dict[temp_key]
        if word in internal_dict:
            internal_dict[word] += 1
        else:
            internal_dict[word] = 1
    return tf_dict, set_of_all_word_vectors

# Intermediate dictionary needed to generate TF-IDF vector
def generate_tfidf_dict(tf_dict):
    tfidf_dict = {}
    for key, value in tf_dict.items():
        sensor_id = key[1]
        if sensor_id not in tfidf_dict:
            tfidf_dict[sensor_id] = set()
        for internal_key in value.keys():
            tfidf_dict[sensor_id].add(internal_key)
    return tfidf_dict

# Intermediate dictionary needed to generate TF-IDF2 vector
def generate_tf_idf2_dict(tf_dict):
    tfidf2_dict = {}
    for key, value in tf_dict.items():
        gesture_id = key[0]
        if gesture_id not in tfidf2_dict:
            tfidf2_dict[gesture_id] = set()
        for internal_key in value.keys():
            tfidf2_dict[gesture_id].add(internal_key)
    return tfidf2_dict

# Generate TF vectors
def generate_tf_vectors(tf_dict, word_vector_pos_dict):
    tf_vectors = {}
    for key, value in tf_dict.items():
        gesture_id = key[0]
        sensor_id = key[1]
        if gesture_id not in tf_vectors:
            tf_vectors[gesture_id] = [0] * len(word_vector_pos_dict.keys())
        tf_list = tf_vectors[gesture_id]
        for internal_key in value.keys():
            tf_list[word_vector_pos_dict[str(internal_key)]] = value[internal_key]
    
    for gesture_id in tf_vectors.keys():
        tf_list = tf_vectors[gesture_id]
        sum = 0
        for x in tf_list:
            sum += x
        tf_list[:] = [x / sum for x in tf_list]
        tf_vectors[gesture_id] = tuple(tf_list)
    return tf_vectors

# Generate IDF vectors
def generate_idf_vectors(tf_dict, tf_idf_dict, word_vector_pos_dict):
    idf_vectors = {}
    total_num_documents = (len(tf_dict.keys()) / len(tf_idf_dict.keys()))
    idf_list = [0] * len(word_vector_pos_dict.keys())
    for sensor_id, complete_word_set in tf_idf_dict.items():
        for word in complete_word_set:
            count = 0
            for key, value in tf_dict.items():
                if key[1] == sensor_id:
                    word_list = list(tf_dict[key].keys())
                    if word in word_list:
                        count += 1
            idf_list[word_vector_pos_dict[str(word)]] = count
    for i in range(0, len(idf_list)):
        if(idf_list[i] == 0):
            continue;
        idf_list[i] = math.log(total_num_documents)/idf_list[i]
    idf_vectors = tuple(idf_list)
    return idf_vectors

# Generate IDF2 vectors
def generate_idf2_vectors(tf_dict, tf_idf2_dict, word_vector_pos_dict):
    idf2_vectors = {}
    total_num_documents = (len(tf_dict.keys()) / len(tf_idf2_dict.keys()))
    for gesture_id, complete_word_set in tf_idf2_dict.items():
        idf_list = [0] * len(word_vector_pos_dict.keys())
        for word in complete_word_set:
            count = 0
            for key, value in tf_dict.items():
                if key[0] == gesture_id:
                    word_list = list(tf_dict[key].keys())
                    if word in word_list:
                        count += 1
            idf_list[word_vector_pos_dict[str(word)]] = count
        for i in range(0, len(idf_list)):
            if(idf_list[i] == 0):
                continue;
            idf_list[i] = math.log(total_num_documents/idf_list[i])
        idf2_vectors[gesture_id] = tuple(idf_list)
    return idf2_vectors

# Combine all three vector representations into onw doctionary
def combine_all_vectors(tf_vectors, idf_vectors, idf2_vectors, word_vector_pos_dict):
    vectors = {}
    for gesture_id in tf_vectors.keys():
        tf_vector_tuple = tf_vectors[gesture_id]
        idf_vector_tuple = idf_vectors
        idf2_vector_tuple = idf2_vectors[gesture_id]
        tfidf_vector_tuple = tuple([a*b for a,b in zip(tf_vector_tuple,idf_vector_tuple)])
        tfidf2_vector_tuple = tuple([a*b for a,b in zip(tf_vector_tuple,idf2_vector_tuple)])
        vector_list = [tf_vector_tuple, tfidf_vector_tuple, tfidf2_vector_tuple]
        vectors[gesture_id] = vector_list
    return vectors

# Serialize vector dictiionary
def serialize_vectors_dictionary(vectors):
    with open("Intermediate/vectors.json", "w") as write_file:
        json.dump(vectors, write_file)

# Serialize dictionary which stores the position of each word in the vector representaitons
def serialize_word_vector_pos_dictionary(word_vector_pos_dict):
    with open("Intermediate/word_vector_pos_dict.json", "w") as write_file:
        json.dump(word_vector_pos_dict, write_file)

# Deserialize data parameters
def deserialize_data_parameters(file_name):
    with open(file_name, "r") as read_file:
        data = json.load(read_file)
    return data

# Delete vectors.txt file if present
def delete_vector_text_file(directory):
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if file.endswith("vectors.txt")]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)

# Write vectors.txt file
def write_vector_text_file(directory, vectors_dict):
    delete_vector_text_file(directory)
    file_path = directory + "/vectors.txt"
    f = open(file_path, "w")
    for key, value in vectors_dict.items():
        f.write(str(key) + ":\n")
        for i in range(0, len(value)):
            if i == 0:
                f.write("\tTF vector :" + str(value[i]) + "\n")
            elif i == 1:
                f.write("\tTF-IDF vector :" + str(value[i]) + "\n")
            else:
                f.write("\tTF-IDF2 vector :" + str(value[i]) + "\n")
    f.close()

def generate_vectors():
    print("Deserializing objects from previous tasks...")
    file_name = "Intermediate/gesture_word_dictionary.json"
    word_vector_dict = deserialize_gesture_word_dictionary(file_name)

    data_parameters_file_name = "Intermediate/data_parameters.json"
    data = deserialize_data_parameters(data_parameters_file_name)

    print("Generating TF vector...")
    tf_dict, set_of_all_word_vectors = generate_tf_dict(word_vector_dict)

    print("Generating TF-IDF vector...")
    tfidf_dict = generate_tfidf_dict(tf_dict)

    print("Generating TF-IDF2 vector...")
    tfidf2_dict = generate_tf_idf2_dict(tf_dict)

    list_of_all_word_vectors = list(set_of_all_word_vectors)

    # Create a dictionary containing the position of each word
    # in the vector representations
    word_vector_pos_dict = {}
    k=0
    for word in list_of_all_word_vectors:
        word_vector_pos_dict[str(word)] = k
        k += 1
    
    # Generate all three vector represenations
    tf_vectors = generate_tf_vectors(tf_dict, word_vector_pos_dict)
    idf_vectors = generate_idf_vectors(tf_dict, tfidf_dict, word_vector_pos_dict)
    idf2_vectors = generate_idf2_vectors(tf_dict, tfidf2_dict, word_vector_pos_dict)

    # Combine all the three representations of all gestures into one dictionary
    print("Writing vectors.txt...")
    vectors_dict = combine_all_vectors(tf_vectors, idf_vectors, idf2_vectors, word_vector_pos_dict)

    # Write vectors.txt
    write_vector_text_file(data['directory'], vectors_dict)

    print("Serializing objects needed for future tasks...")
    serialize_vectors_dictionary(vectors_dict)
    serialize_word_vector_pos_dictionary(word_vector_pos_dict)

    print("Task-2 complete!")

def main():
    while(True):
        print("\n******************** Task-2 **********************")
        print(" Enter 1 to generate TF, TF-IDF and TF-IDF2 vectors.")
        print(" Enter 2 to exit")
        option = input("Enter option: ")
        if option == '1':
            task1.create_output_directories()
            generate_vectors()
        else:
            break

if __name__ == "__main__":
    main()
