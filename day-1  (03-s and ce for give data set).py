import csv
import numpy as np
import pandas as pd

def load_data(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        for row in csv.reader(csvfile):
            data.append(row)
    return data

def initialize_hypothesis(data):
    num_attributes = len(data[0]) - 1
    return ['0'] * num_attributes

def find_s_algorithm(data):
    hypothesis = initialize_hypothesis(data)
    for instance in data:
        if instance[-1] == 'yes':
            for j in range(len(instance) - 1):
                if hypothesis[j] == '0' or hypothesis[j] == instance[j]:
                    hypothesis[j] = instance[j]
                else:
                    hypothesis[j] = '?'
    return hypothesis

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h

def main():
    file_path = "C:\machine learning\Book11.csv"
    data = load_data(file_path)
    print("The total number of training instances are:", len(data))
    
    # Applying FIND-S Algorithm
    print("\nFIND-S Algorithm:")
    hypothesis_find_s = find_s_algorithm(data)
    print("The initial hypothesis is:", hypothesis_find_s)
    
    # Applying Candidate Elimination Algorithm
    print("\nCandidate Elimination Algorithm:")
    data_pd = pd.DataFrame(data=pd.read_csv(file_path))
    concepts = np.array(data_pd.iloc[:, 0:-1])
    target = np.array(data_pd.iloc[:, -1])
    specific_hypothesis, general_hypotheses = learn(concepts, target)
    print("Final Specific_h:", specific_hypothesis)
    print("Final General_h:", general_hypotheses)

if __name__ == "__main__":
    main()
