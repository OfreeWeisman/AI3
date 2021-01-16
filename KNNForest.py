import pandas as pd
import numpy as np
import random
import copy
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

N = 10
K = 3
p = 0.6


class KNNForest:
    def __init__(self, forest=None, centroids=None):
        self.forest = forest
        self.centroids = centroids

    def fit(self, patients):
        """ This function splits the training data to N different groups
            For each group, builds a decision tree and centroid vector"""

        forest = []
        centroids = []

        # split the training data to N groups of patients
        # For each group built tree and calculate the centroid
        if patients is None:
            patients = self.tableToPatients('train.csv')
        for i in range(N):
            patients_group = self.randomPatientsGroup(patients, p)
            tree = self.buildTree(patients_group)
            centroid = self.calcCentroid(patients_group)
            forest.append(tree)
            centroids.append(centroid)

        self.forest = forest
        self.centroids = centroids

    def predict(self, patients):
        """ For each patient in the test data, choose K trees from the forest
            Return the majority's decision
        """
        if patients is None:
            patients = self.tableToPatients('test.csv')
        counter = 0
        # for each patient find K trees with the closest centroid to the patients
        for patient in patients:
            committee = self.chooseKTrees(patient.symptoms)
            sick = 0
            healthy = 0
            for tree in committee:
                if self.diagnose(tree, patient) == 'M':
                    sick += 1
                else:
                    healthy += 1

            decision = 'B' if healthy > sick else 'M'
            if decision == patient.diagnosis:
                counter += 1

        total = len(patients)
        return counter / total

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

    def diagnose(self, node, patient):
        if node.decision is not None:
            return node.decision

        symptom_val = patient.symptoms[node.feature]
        if symptom_val < node.threshold:
            return self.diagnose(node.left_sub_tree, patient)
        else:
            return self.diagnose(node.right_sub_tree, patient)

    def calcCentroid(self, patients):
        num = len(patients)
        length = len(patients[0].symptoms)
        centroid = []
        for i in range(length):
            val = 0
            for patient in patients:
                val += patient.symptoms[i]
            centroid.append(val / num)
        return centroid

    def chooseKTrees(self, vector):
        k_trees = []
        my_centroids = copy.deepcopy(self.centroids)
        for i in range(K):
            closest = self.closestVector(vector, my_centroids)
            index = self.getCentroidIndex(closest)
            k_trees.append(self.forest[index])
            my_centroids.remove(closest)
        return k_trees

    def getCentroidIndex(self, c):
        index = 0
        for i in range(len(self.centroids)):
            if self.centroids[i] != c:
                index += 1
            else:
                return index

    def closestVector(self, vector, vectors):
        min_dist = float('inf')
        best_v = None
        for v in vectors:
            dist = self.distanceBetweenVectors(vector, v)
            if dist < min_dist:
                min_dist = dist
                best_v = v
        return best_v

    def distanceBetweenVectors(self, v1, v2):
        diff = [np.abs(v1[i] - v2[i]) for i in range(len(v1))]
        diff = sum(list(map(lambda x: x * x, diff)))
        return diff

    def buildTree(self, patients):
        # Create the root node of the decision tree
        root = self.Node(None, None, None, patients, None, None)
        # Recursively build the tree and update the class field
        self.splitNode(root)
        return root

    def randomPatientsGroup(self, patients, param):
        group = []
        num = int(len(patients) * param)
        patients_list = list(np.copy(patients))
        for i in range(num):
            patient = random.choice(patients_list)
            group.append(patient)
            patients_list.remove(patient)
        return group

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
    knn = KNNForest()
    kf = KFold(n_splits=5, shuffle=True, random_state=123456789)
    data_frame = pd.read_csv('train.csv')
    table = data_frame.values.tolist()
    # ns = [5, 10, 15, 20]
    # ks = [3, 5, 7, 11, 18]
    # ps = [0.3, 0.4, 0.5, 0.6, 0.7]
    ns = [8, 16, 24]
    ks = [3, 5, 9, 11]
    ps = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    experiment_results = []
    for n in ns:
        for k in ks:
            if k >= n:
                break
            for p in ps:
                N = n
                K = k
                results = 0
                for train_index, test_index in kf.split(table):
                    patients_for_train = dataToPatients(knn, train_index)
                    patients_for_test = dataToPatients(knn, test_index)
                    knn.fit(patients_for_train)
                    results += knn.predict(patients_for_test)
                average_success_rate = results / 5
                experiment_results.append((n, k, p, average_success_rate))
                print((k, n, p, average_success_rate))
    print(experiment_results)


if __name__ == '__main__':
    # experiments()
    ps = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    for i in range(10):
        for param in ps:
            p = param
            knn = KNNForest()
            knn.fit(None)
            print('p=', p, 'accuracy=', knn.predict(None))
    # ns = [5, 10, 15, 20]
    # ks = [3, 7, 11, 18]
    # ns = [5, 10]
    # ks = [3, 5, 7]
    # ps = [0.3, 0.4, 0.5, 0.6, 0.7]
    #
    # best_n = None
    # best_k = None
    # best_p = None
    # acc = 0
    #
    # for n in ns:
    #     for k in ks:
    #         if k > n:
    #             break
    #         N = n
    #         K = k
    #         # accuracies = []
    #         for param in ps:
    #             p = param
    #             knn.fit()
    #             accuracy = knn.predict()
    #             # accuracies.append(accuracy)
    #             if accuracy > acc:
    #                 best_n = n
    #                 best_k = k
    #                 best_p = param
    #                 acc = accuracy
    #         print('N=', n, 'K=', k, 'accuracy=', accuracy)
    #
    #         # plt.plot(ps, accuracies)
    #         # plt.xlabel('Parameters')
    #         # plt.ylabel('Accuracy')
    #         # print('N=', n, 'K=', k)
    #         # plt.show()
    # print('accuracy=', acc, 'N=', best_n, 'K=', best_k, 'p=', best_p)
