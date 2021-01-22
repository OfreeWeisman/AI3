import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class ID3:
    def __init__(self,
                 decision_tree=None, param=None):
        self.param = param
        self.decision_tree = decision_tree

    def fit(self, patients):
        """ This function builds the decision tree according to the training data"""
        if patients is None:
            # Create the patients collection from the training data
            patients = self.tableToPatients('train.csv')
        # Create the root node of the decision tree
        root = self.Node(None, None, None, patients, None, None)
        # Recursively build the tree and update the class field
        self.improvedSplitNode(root)
        self.decision_tree = root

    def predict(self, patients):
        """ This function receives an item to inspect and decides whether the result is 'ill' or 'health' """
        if patients is None:
            patients = self.tableToPatients('test.csv')
        counter = 0
        fp = 0
        fn = 0
        for p in patients:
            result, decision = self.diagnose(self.decision_tree, p)
            if result:
                counter += 1
            if decision == 'M' and p.diagnosis == 'B':
                fp += 1
            if decision == 'B' and p.diagnosis == 'M':
                fn += 1

        total = len(patients)
        loss = ((0.1 * fp) + fn) / total
        return loss

    def diagnose(self, node, patient):
        if node.decision is not None:
            return patient.diagnosis == node.decision, node.decision

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
        return - (p_healthy * np.log2(p_healthy)) - (p_sick * np.log2(p_sick))

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
        values = [p.symptoms[feature] for p in patients]
        values = sorted(values)
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
            if ig > best_ig:
                best_ig = ig
                threshold = mid
                less_than = smaller
                greater_than = bigger

        return best_ig, threshold, less_than, greater_than

    def find_feature_and_threshold_to_split_by(self, patients):
        """ This function finds from all the features the best feature that with the right threshold value will provide the best IG
            It is also calculates that threshold value
        """

        num_of_features = len(patients[0].symptoms)

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
            symptoms = list(map(lambda s: float(s), symptoms))
            p = self.Patient(diagnosis, symptoms)
            patients.append(p)
        return patients

    def allSame(self, patients, classifier):
        for p in patients:
            if p.diagnosis != classifier:
                return False
        return True

    def improvedSplitNode(self, node):
        patients = node.patients
        if self.mostSick(patients):
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

        self.improvedSplitNode(node.left_sub_tree)
        self.improvedSplitNode(node.right_sub_tree)

    def mostSick(self, patients):
        total = len(patients)
        sick = 0
        for p in patients:
            if p.diagnosis == 'M':
                sick += 1
        if sick / total >= self.param:
            return True
        return False


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
    kf = KFold(n_splits=5, shuffle=True, random_state=312461270)
    data_frame = pd.read_csv('train.csv')
    table = data_frame.values.tolist()
    parameters = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    experiment_results = []
    for m in parameters:
        results = 0
        id3 = ID3(param=m)
        for train_index, test_index in kf.split(table):
            patients_for_train = dataToPatients(id3, train_index)
            patients_for_test = dataToPatients(id3, test_index)
            id3.fit(patients_for_train)
            results += id3.predict(patients_for_test)
        average_success_rate = results / 5
        experiment_results.append((m, average_success_rate))

    print(experiment_results)


if __name__ == '__main__':
    my_id3 = ID3(param=0.9)
    my_id3.fit(None)
    print(my_id3.predict(None))
    # experiments()
