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
import task3
import task1

# Calculate distance between vector representations of source ans target gestures
def get_distance_from_gesture(source_gesture_id, target_gesture_id, vectors_dict, vector_type):
    total_distance = 0
    source_tf_vector =  vectors_dict[str(source_gesture_id)][vector_type]
    target_tf_vector =  vectors_dict[str(target_gesture_id)][vector_type]
    for i in range(0, len(source_tf_vector)):
        total_distance += pow((source_tf_vector[i]-target_tf_vector[i]), 2)
    return total_distance

# Read user input
def load_gesture_id():
    gesture_id = int(input("Enter gesture id: "))
    print("Enter 1 for TF vector")
    print("Enter 2 for TF-IDF vector")
    print("Enter 3 for TF-IDF2 vector")
    vector_type = int(input())
    # gesture_id = 1
    # vector_type = 1
    return gesture_id, vector_type

# Print similar vectors
def find_similar_vectors():
    gesture_id, vector_type = load_gesture_id()

    print("Deserializing objects from previous tasks...")
    vectors_dictionary_file_name = "Intermediate/vectors.json"
    vectors_dict = task3.deserialize_vectors_dictionary(vectors_dictionary_file_name)

    gesture_ids = pd.read_csv('Intermediate/quantized.csv').gesture_id.unique()

    print("Calculating similarity with other gestures...")
    result = []
    source_gesture_id = gesture_id
    for target_gesture_id in gesture_ids:
        distance = get_distance_from_gesture(source_gesture_id, target_gesture_id, vectors_dict, vector_type-1)
        result.append((target_gesture_id, distance))

    result.sort(key=(lambda b: b[1]))
    fin = []
    for x in result[:10]:
        fin.append(x[0])

    print("Similar gestures in ascending order...")
    print(fin)
    print("Task-4 complete!")

def main():
    while(True):
        print("\n******************** Task-4 **********************")
        print(" Enter 1 to find 10 most similar gestures")
        print(" Enter 2 to exit")
        option = input("Enter option: ")
        if option == '1':
            task1.create_output_directories()
            find_similar_vectors()
        else:
            break

if __name__ == "__main__":
    main()






