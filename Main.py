from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def main():
    student_path = 'data/student-mat.csv'

    test1_path = 'data/student-mat-test-1.csv'
    accuracy_test1 = get_accuracy_from_csv(test1_path, student_path)
    print('Accuracy Test 1 =', accuracy_test1)

    test2_path = 'data/student-mat-test-2.csv'
    accuracy_test2 = get_accuracy_from_csv(test2_path, student_path)
    print('Accuracy Test 2 =', accuracy_test2)

    test3_path = 'data/student-mat-test-3.csv'
    accuracy_test3 = get_accuracy_from_csv(test3_path, student_path)
    print('Accuracy Test 3 =', accuracy_test3)

    test4_path = 'data/student-mat-test-4.csv'
    accuracy_test4 = get_accuracy_from_csv(test4_path, student_path)
    print('Accuracy Test 4 =', accuracy_test4)


def get_accuracy_from_csv(attributes_path, data_path):
    X_test, X_train, y_test, y_train = train_test_data_from_csv(attributes_path, data_path)
    accuracy = get_knn_accuracy(X_test, X_train, y_test, y_train, 6)
    return accuracy


def get_knn_accuracy(X_test, X_train, y_test, y_train, k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy


def train_test_data_from_csv(attributes_path, data_path):
    df_data = pd.read_csv(data_path, sep=';')
    df_attributes = pd.read_csv(attributes_path, sep=';')
    data, target = read_enabled_attributes(df_attributes, df_data, 'result')
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0, shuffle=True)
    return X_test, X_train, y_test, y_train


def read_enabled_attributes(df_attributes, df_data, target_column):
    enabled_attributes = df_attributes.where(df_attributes['enabled']).dropna(thresh=2)
    data = df_data[enabled_attributes['attribute']]
    target = df_data[target_column]
    encode_categorical_data(data)
    return data.values, target.values


def encode_categorical_data(data):
    categorical_feature_mask = data.dtypes == object
    categorical_cols = data.columns[categorical_feature_mask].tolist()
    le = LabelEncoder()

    if len(categorical_cols) > 0:
        data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))


if __name__ == '__main__':
    main()
