import pandas as pd
import numpy as np
import random
import copy
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

N = 15
K = 11
p = 0.6


def distanceBetweenVectors(v1, v2, features):
    diff = [np.abs(v1[i] - v2[i]) for i in features]
    diff = sum(list(map(lambda x: x * x, diff)))
    return diff


def randomPatientsGroup(patients, param):
    group = []
    num = int(len(patients) * param)
    patients_list = list(np.copy(patients))
    for i in range(num):
        patient = random.choice(patients_list)
        group.append(patient)
        patients_list.remove(patient)
    return group


def entropy(patients):
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


def calculateIG(patients, list1, list2):
    h = entropy(patients)
    h1 = entropy(list1)
    h2 = entropy(list2)
    size = len(patients)
    size1 = len(list1)
    size2 = len(list2)
    return h - (((size1 / size) * h1) + ((size2 / size) * h2))


def find_threshold(feature, patients):
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
        ig = calculateIG(patients, smaller, bigger)
        if ig > best_ig:
            best_ig = ig
            threshold = mid
            less_than = smaller
            greater_than = bigger

    return best_ig, threshold, less_than, greater_than


def find_feature_and_threshold_to_split_by(patients):
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
        ig, val, smaller, bigger = find_threshold(i, patients)
        if ig >= best_ig:
            best_ig = ig
            best_feature = i
            best_threshold = val
            final_smaller = smaller
            final_bigger = bigger

    return best_feature, best_threshold, final_smaller, final_bigger


def allSame(patients, classifier):
    for p in patients:
        if p.diagnosis != classifier:
            return False
    return True


def calcCentroid(patients):
    num = len(patients)
    length = len(patients[0].symptoms)
    centroid = []
    for i in range(length):
        val = 0
        for patient in patients:
            val += patient.symptoms[i]
        centroid.append(val / num)
    return centroid


class KNNForest:
    def __init__(self, forest=None, centroids=None, norForest=None, norCentroids=None, minmax_values=None, param=None, features=None, nor_features=None):
        self.nor_features = nor_features
        self.features = features
        self.param = param
        self.minmax_values = minmax_values
        self.norForest = norForest
        self.norCentroids = norCentroids
        self.forest = forest
        self.centroids = centroids

    def fit(self, patients):
        """ This function splits the training data to N different groups
            For each group, builds a decision tree and centroid vector"""

        forest = []
        centroids = []
        normalized_forest = []
        normalized_centroids = []
        features = set([])
        nor_features = set([])

        # split the training data to N groups of patients
        # For each group built tree and calculate the centroid
        if patients is None:
            patients = self.tableToPatients('train.csv')

        for i in range(N):
            patients_group = randomPatientsGroup(patients, p)
            normalized_group = self.normalize(patients_group)
            tree, selected_features = self.buildTree(patients_group)
            features = features | set(selected_features)
            centroid = calcCentroid(patients_group)
            normalized_tree, nor_selected_features = self.buildTree(normalized_group)
            nor_features = nor_features | set(nor_selected_features)
            normalized_centroid = calcCentroid(normalized_group)
            forest.append(tree)
            centroids.append(centroid)
            normalized_forest.append(normalized_tree)
            normalized_centroids.append(normalized_centroid)

        self.forest = forest
        self.centroids = centroids
        self.norForest = normalized_forest
        self.norCentroids = normalized_centroids
        self.features = features
        self.nor_features = nor_features

    def predict(self, patients):
        """ For each patient in the test data, choose K trees from the forest
            Return the majority's decision
        """
        if patients is None:
            patients = self.tableToPatients('test.csv')
        d_counter = 0
        d_normalized_counter = 0
        counter = 0
        normalized_counter = 0
        # for each patient find K trees with the closest centroid to the patients
        for patient in patients:
            normalized_symptoms = copy.deepcopy(patient.symptoms)
            length = len(normalized_symptoms)
            normalized_symptoms_vals = [(normalized_symptoms[i] - self.minmax_values[i][0]) / (
                    self.minmax_values[i][1] - self.minmax_values[i][0]) for i in range(length)]
            d_committee, d_normalized_committee = self.chooseKTreesDynamically(patient.symptoms,
                                                                               normalized_symptoms_vals)
            committee, normalized_committee = self.chooseKTrees(patient.symptoms, normalized_symptoms_vals)
            sick = 0
            healthy = 0
            for tree in d_committee:
                if self.diagnose(tree, patient) == 'M':
                    sick += 1
                else:
                    healthy += 1

            decision = 'B' if healthy > sick else 'M'
            if decision == patient.diagnosis:
                d_counter += 1

            sick = 0
            healthy = 0
            n_patient = self.normFeatures(patient)
            for tree in d_normalized_committee:
                if self.diagnose(tree, n_patient) == 'M':
                    sick += 1
                else:
                    healthy += 1

            decision = 'B' if healthy > sick else 'M'
            if decision == n_patient.diagnosis:
                d_normalized_counter += 1

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

            sick = 0
            healthy = 0
            n_patient = self.normFeatures(patient)
            for tree in normalized_committee:
                if self.diagnose(tree, n_patient) == 'M':
                    sick += 1
                else:
                    healthy += 1

            decision = 'B' if healthy > sick else 'M'
            if decision == n_patient.diagnosis:
                normalized_counter += 1

        total = len(patients)
        return d_counter / total, d_normalized_counter / total, counter / total, normalized_counter / total

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
                     right_sub_tree, selected_features):
            self.selected_features = selected_features
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

    def closestVector(self, vector, vectors, features):
        min_dist = float('inf')
        best_v = None
        for v in vectors:
            dist = distanceBetweenVectors(vector, v, features)
            if dist < min_dist:
                min_dist = dist
                best_v = v
        return best_v

    def chooseKTrees(self, vector, nvector):
        k_trees = []
        normalized_k_trees = []
        my_centroids = copy.deepcopy(self.centroids)
        my_normalized_centroids = copy.deepcopy(self.norCentroids)
        for i in range(K):
            closest = self.closestVector(vector, my_centroids, self.features)
            nor_closest = self.closestVector(nvector, my_normalized_centroids, self.nor_features)
            index = self.getCentroidIndex(closest)
            nor_index = self.getNormalizedCentroidIndex(nor_closest)
            k_trees.append(self.forest[index])
            normalized_k_trees.append(self.norForest[nor_index])
            my_centroids.remove(closest)
            my_normalized_centroids.remove(nor_closest)
        return k_trees, normalized_k_trees

    def chooseKTreesDynamically(self, vector, nvector):
        k_trees = []
        normalized_k_trees = []
        my_centroids = copy.deepcopy(self.centroids)
        my_normalized_centroids = copy.deepcopy(self.norCentroids)

        # find closes centroid and tree
        closest = self.closestVector(vector, my_centroids, self.features)
        nor_closest = self.closestVector(nvector, my_normalized_centroids, self.nor_features)
        index = self.getCentroidIndex(closest)
        nor_index = self.getNormalizedCentroidIndex(nor_closest)
        k_trees.append(self.forest[index])
        normalized_k_trees.append(self.norForest[nor_index])
        my_centroids.remove(closest)
        my_normalized_centroids.remove(nor_closest)

        # calc max distance
        max_dist = self.param * distanceBetweenVectors(vector, closest, self.features)
        n_max_dist = self.param * distanceBetweenVectors(nvector, nor_closest, self.features)

        # find all centroids and trees which are closer than max_dist
        for i in range(len(my_centroids)):
            closest = self.closestVector(vector, my_centroids, self.features)
            if distanceBetweenVectors(vector, closest, self.features) <= max_dist:
                index = self.getCentroidIndex(closest)
                k_trees.append(self.forest[index])
                my_centroids.remove(closest)
            else:
                break
        if len(k_trees) % 2 == 0:
            k_trees.pop()

        for j in range(len(my_normalized_centroids)):
            nor_closest = self.closestVector(nvector, my_normalized_centroids, self.nor_features)
            dist = distanceBetweenVectors(nvector, nor_closest, self.features)
            if dist <= n_max_dist:
                nor_index = self.getNormalizedCentroidIndex(nor_closest)
                normalized_k_trees.append(self.norForest[nor_index])
                my_normalized_centroids.remove(nor_closest)
            else:
                break
        if len(normalized_k_trees) % 2 == 0:
            normalized_k_trees.pop()

        return k_trees, normalized_k_trees

    def getCentroidIndex(self, c):
        index = 0
        for i in range(len(self.centroids)):
            if self.centroids[i] != c:
                index += 1
            else:
                return index

    def getNormalizedCentroidIndex(self, c):
        index = 0
        for i in range(len(self.norCentroids)):
            if self.norCentroids[i] != c:
                index += 1
            else:
                return index

    def buildTree(self, patients):
        # Create the root node of the decision tree
        selected_features = []
        root = self.Node(None, None, None, patients, None, None, selected_features)
        # Recursively build the tree and update the class field
        self.splitNode(root)
        return root, selected_features

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

    def splitNode(self, node):
        patients = node.patients
        if allSame(patients, 'M'):
            node.decision = 'M'
            return
        elif allSame(patients, 'B'):
            node.decision = 'B'
            return

        feature, threshold, smaller, bigger = find_feature_and_threshold_to_split_by(patients)
        node.selected_features.append(feature)
        left_son = self.Node(None, None, None, smaller, None, None, node.selected_features)
        right_son = self.Node(None, None, None, bigger, None, None, node.selected_features)
        node.left_sub_tree = left_son
        node.right_sub_tree = right_son
        node.threshold = threshold
        node.feature = feature

        self.splitNode(node.left_sub_tree)
        self.splitNode(node.right_sub_tree)

    def normalize(self, patients):
        minmax = []
        num_of_features = len(patients[0].symptoms)
        norm = copy.deepcopy(patients)
        for i in range(num_of_features):
            feature_values = [patient.symptoms[i] for patient in norm]
            min_val = min(feature_values)
            max_val = max(feature_values)
            minmax.append((min_val, max_val))
            for patient in norm:
                normalized_val = (patient.symptoms[i] - min_val) / (max_val - min_val)
                patient.symptoms[i] = normalized_val
        self.minmax_values = minmax
        return norm

    def normFeatures(self, patient):
        num_of_features = len(patient.symptoms)
        norm_patient = copy.deepcopy(patient)
        for i in range(num_of_features):
            feature_value = (norm_patient.symptoms[i] - (self.minmax_values[i])[0]) / (
                    (self.minmax_values[i])[1] - (self.minmax_values[i])[0])
            if feature_value < 0:
                feature_value = 0
            elif feature_value > 1:
                feature_value = 1
            norm_patient.symptoms[i] = feature_value
        return norm_patient


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


def removeFeatures(num_of_features):
    features = list(range(num_of_features))

    kf = KFold(n_splits=5, shuffle=True, random_state=123456789)
    data_frame = pd.read_csv('train.csv')
    table = data_frame.values.tolist()
    results = 0
    for train_index, test_index in kf.split(table):
        patients_for_train = dataToPatients(knn, train_index)
        patients_for_test = dataToPatients(knn, test_index)
        knn.fit(patients_for_train)
        results += knn.predict(patients_for_test)[0]
    best_precision = results / 5

    while len(features) > 0:
        f_to_remove = None
        for f in features:
            print('f = ', f)
            new_features = copy.deepcopy(features)
            new_features.remove(f)
            results = 0
            for train_index, test_index in kf.split(table):
                patients_for_train = dataToPatients(knn, train_index)
                for p1 in patients_for_train:
                    p1.symptoms.remove(p1.symptoms[f])
                patients_for_test = dataToPatients(knn, test_index)
                for p2 in patients_for_test:
                    p2.symptoms.remove(p2.symptoms[f])
                knn.fit(patients_for_train)
                results += knn.predict(patients_for_test)[0]
            if results / 5 > best_precision:
                best_precision = results / 5
                f_to_remove = f
        if f_to_remove is None:
            break
        features.remove(f_to_remove)
    return features


def experiments():
    kf = KFold(n_splits=5, shuffle=True, random_state=312461270)
    data_frame = pd.read_csv('train.csv')
    table = data_frame.values.tolist()
    parameters = [1.6, 1.8, 2, 2.2, 3, 3.5, 4, 4.5, 5]
    experiment_results = []
    best = 0, 0
    for m in parameters:
        knn = KNNForest(param=m)
        results = 0
        for train_index, test_index in kf.split(table):
            patients_for_train = dataToPatients(knn, train_index)
            patients_for_test = dataToPatients(knn, test_index)
            knn.fit(patients_for_train)
            results += knn.predict(patients_for_test)[0]
        average_success_rate = results / 5
        if average_success_rate > best[1]:
            best = (m, average_success_rate)
        experiment_results.append((m, average_success_rate))
        print((m, average_success_rate))
    print(experiment_results)
    print('best:', best)


if __name__ == '__main__':
    knn = KNNForest(param=1.5)
    knn.fit(None)
    d_regular, d_normalized, regular, normalized = knn.predict(None)
    print(d_regular)
