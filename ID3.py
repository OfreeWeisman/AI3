import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class ID3:
    def __init__(self,
                 decision_tree=None, m=0):
        self.decision_tree = decision_tree
        self.m = m

    def fit(self, patients):
        """ This function builds the decision tree according to the training data"""
        if patients is None:
            # Create the patients collection from the training data
            patients = self.tableToPatients('train.csv')
        # Create the root node of the decision tree
        root = self.Node(None, None, None, patients, None, None, None)
        # Recursively build the tree and update the class field
        if self.m == 0:
            self.splitNode(root)
        else:
            self.mSplitNode(root, self.m)
        self.decision_tree = root

    def predict(self, patients):
        """ This function receives an item to inspect and decides whether the result is 'ill' or 'health' """
        if patients is None:
            patients = self.tableToPatients('test.csv')
        counter = 0
        for p in patients:
            if self.diagnose(self.decision_tree, p):
                counter += 1

        total = len(patients)
        return counter / total

    def diagnose(self, node, patient):
        if node.decision is not None:
            return patient.diagnosis == node.decision

        symptom_val = patient.symptoms[node.feature]
        if symptom_val < node.threshold:
            return self.diagnose(node.left_sub_tree, patient)
        else:
            return self.diagnose(node.right_sub_tree, patient)

    class Node:
        """ This class represents a node of the decision tree.
            Each node has a feature and threshold value to decide whether an example represents ill patient or healthy patient
        """

        def __init__(self,
                     feature,
                     threshold,
                     decision,
                     patients,
                     left_sub_tree,
                     right_sub_tree,
                     fathers_patients):
            self.feature = feature
            self.threshold = threshold
            self.decision = decision
            self.patients = patients
            self.left_sub_tree = left_sub_tree
            self.right_sub_tree = right_sub_tree
            self.fathers_patients = fathers_patients

    class Patient:
        """ This class represents one example = one line in the table
            Symptoms is a collection of tuples : (feature, value)
        """

        def __init__(self,
                     diagnosis,
                     symptoms):
            self.diagnosis = diagnosis
            self.symptoms = symptoms

    def entropy(self, patients):
        sick = 0
        healthy = 0
        total = len(patients)
        if total == 0:
            return 0
        for p in patients:
            if p.diagnosis == 'M':
                sick += 1
            else:
                healthy += 1
        p_sick = sick / total
        p_healthy = healthy / total
        if p_sick == 0 or p_healthy == 0:
            return 0
        return - (p_healthy * np.log(p_healthy)) - (p_sick * np.log(p_sick))

    def calculateIG(self, patients, list1, list2):
        h = self.entropy(patients)
        h1 = self.entropy(list1)
        h2 = self.entropy(list2)
        size = len(patients)
        size1 = len(list1)
        size2 = len(list2)
        return h - (((size1 / size) * h1) + ((size2 / size) * h2))

    def find_threshold(self, feature, patients):
        """
        This function split the values in a way that provides the best IG
        The 'feature' argument is an index of the value in the symptoms list of every patient
        """

        if len(patients) <= 1:
            return 0, None, None, None
        data_frame = pd.read_csv('train.csv')
        table = data_frame.values.tolist()
        values = []
        for i in range(0, len(table)):
            values.append(table[i][feature + 1])

        values.sort()
        best_ig = 0
        threshold = None
        less_than = []
        greater_than = []

        # for each mid value, split the list and calc the IG. return the value that provides the greatest IG
        for i in range(len(values) - 1):
            mid = (values[i] + values[i + 1]) / 2
            smaller = []
            bigger = []
            for p in patients:
                if feature >= len(p.symptoms):
                    print(feature)
                if p.symptoms[feature] < mid:
                    smaller.append(p)
                else:
                    bigger.append(p)
            ig = self.calculateIG(patients, smaller, bigger)
            if ig >= best_ig:
                best_ig = ig
                threshold = mid
                less_than = smaller
                greater_than = bigger

        return best_ig, threshold, less_than, greater_than

    def find_feature_and_threshold_to_split_by(self, patients):
        """ This function finds from all the features the best feature that with the right threshold value will provide the best IG
            It is also calculates that threshold value
        """

        # Create a list of all features
        examples = pd.read_csv("train.csv")
        features = list(examples.columns)
        num_of_features = len(features) - 1

        # For each feature find the value to split by that provides the best IG
        # Save the best feature and the threshold
        best_ig = 0
        best_feature = None
        best_threshold = None
        final_smaller = []
        final_bigger = []
        for i in range(num_of_features):
            ig, val, smaller, bigger = self.find_threshold(i, patients)
            if ig >= best_ig:
                best_ig = ig
                best_feature = i
                best_threshold = val
                final_smaller = smaller
                final_bigger = bigger

        return best_feature, best_threshold, final_smaller, final_bigger

    def tableToPatients(self, table):
        patients = []
        data_frame = pd.read_csv(table)
        table = data_frame.values.tolist()

        rows = len(table)
        for i in range(rows):
            diagnosis = table[i][0]
            symptoms = table[i]
            symptoms.pop(0)
            p = self.Patient(diagnosis, symptoms)
            patients.append(p)
        return patients

    def allSame(self, patients, classifier):
        for p in patients:
            if p.diagnosis != classifier:
                return False
        return True

    def splitNode(self, node):
        patients = node.patients
        if self.allSame(patients, 'M'):
            node.decision = 'M'
            return
        elif self.allSame(patients, 'B'):
            node.decision = 'B'
            return

        feature, threshold, smaller, bigger = self.find_feature_and_threshold_to_split_by(patients)
        left_son = self.Node(None, None, None, smaller, None, None, patients)
        right_son = self.Node(None, None, None, bigger, None, None, patients)
        node.left_sub_tree = left_son
        node.right_sub_tree = right_son
        node.threshold = threshold
        node.feature = feature

        self.splitNode(node.left_sub_tree)
        self.splitNode(node.right_sub_tree)

    def majority(self, patients):
        sick = 0
        healthy = 0
        for p in patients:
            if p.diagnosis == 'M':
                sick += 1
            else:
                healthy += 1
        if sick >= healthy:
            return 'M'
        return 'B'

    def mSplitNode(self, node, m):
        patients = node.patients
        if len(patients) < m:
            node.decision = self.majority(node.fathers_patients)
            return
        elif self.allSame(patients, 'M'):
            node.decision = 'M'
            return
        elif self.allSame(patients, 'B'):
            node.decision = 'B'
            return

        feature, threshold, smaller, bigger = self.find_feature_and_threshold_to_split_by(patients)
        left_son = self.Node(None, None, None, smaller, None, None, patients)
        right_son = self.Node(None, None, None, bigger, None, None, patients)
        node.left_sub_tree = left_son
        node.right_sub_tree = right_son
        node.threshold = threshold
        node.feature = feature
        self.mSplitNode(node.left_sub_tree, m)
        self.mSplitNode(node.right_sub_tree, m)


def dataToPatients(id3, indices):
    patients = []
    data_frame = pd.read_csv('train.csv')
    table = data_frame.values.tolist()
    patients_table = [table[i] for i in indices]

    rows = len(patients_table)
    for i in range(rows):
        diagnosis = patients_table[i][0]
        symptoms = patients_table[i]
        symptoms.pop(0)
        p = id3.Patient(diagnosis, symptoms)
        patients.append(p)
    return patients


def experiments():
    kf = KFold(n_splits=5, shuffle=True, random_state=123456789)
    data_frame = pd.read_csv('train.csv')
    table = data_frame.values.tolist()
    # parameters = [1, 5, 7, 12, 15, 50, 100, 200]
    # parameters = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
    # parameters = [2, 5, 7, 12, 30, 80, 120]
    parameters = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
    experiment_results = []
    for m in parameters:
        results = 0
        id3 = ID3(m=m)
        for train_index, test_index in kf.split(table):
            patients_for_train = dataToPatients(id3, train_index)
            patients_for_test = dataToPatients(id3, test_index)
            id3.fit(patients_for_train)
            results += id3.predict(patients_for_test)
        average_success_rate = results / 5
        experiment_results.append(average_success_rate)
    print(experiment_results)

    plt.plot(parameters, experiment_results)
    plt.xlabel('Parameter M')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    # my_id3 = ID3(m=7)
    # my_id3.fit(None)
    # print(my_id3.predict(None))

    # my_id3 = ID3(None)
    # my_id3.fit(None)
    # print(my_id3.predict(None))
    experiments()
