import pandas as pd
import numpy as np


class ID3:
    def __init__(self,
                 decision_tree):
        self.decision_tree = decision_tree

    def fit(self):
        """ This function builds the decision tree according to the training data"""

        # Create the patients collection from the training data
        patients = self.tableToPatients('train.csv')
        # Create the root node of the decision tree
        root = self.Node(None, None, None, patients, None, None)
        # Recursively build the tree and update the class field
        self.splitNode(root)
        self.decision_tree = root
        # print(root.feature)
        # print(len(root.left_sub_tree.patients))
        # print(len(root.right_sub_tree.patients))

    def predict(self):
        """ This function receives an item to inspect and decides whether the result is 'ill' or 'health' """
        patients = self.tableToPatients('test.csv')
        counter = 0
        for p in patients:
            if self.diagnose(self.decision_tree, p):
                counter += 1
        total = len(patients)
        print(counter / total)

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
                     right_sub_tree):
            self.feature = feature
            self.threshold = threshold
            self.decision = decision
            self.patients = patients
            self.left_sub_tree = left_sub_tree
            self.right_sub_tree = right_sub_tree

    class Patient:
        """ This class represents one example = one line in the table
            Symptoms is a collection of tuples : (feature, value)
        """

        def __init__(self,
                     pid,
                     diagnosis,
                     symptoms):
            self.pid = pid
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

        data_frame = pd.read_csv('train.csv')
        table = data_frame.values.tolist()
        values = []
        for i in range(1, len(table)):
            values.append(table[i][feature + 1])

        values = list(map(lambda s: float(s), values))
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
        for i in range(1, rows):
            diagnosis = table[i][0]
            symptoms = table[i]
            symptoms.pop(0)
            symptoms = list(map(lambda s: float(s), symptoms))
            p = self.Patient(i + 1, diagnosis, symptoms)
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
        left_son = self.Node(None, None, None, smaller, None, None)
        right_son = self.Node(None, None, None, bigger, None, None)
        node.left_sub_tree = left_son
        node.right_sub_tree = right_son
        node.threshold = threshold
        node.feature = feature

        self.splitNode(node.left_sub_tree)
        self.splitNode(node.right_sub_tree)


if __name__ == '__main__':
    id3 = ID3(None)
    id3.fit()
    id3.predict()
