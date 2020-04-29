from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from StudentClassifier import StudentClassifier
import pandas as pd

algorithm = 'gradientBoosting'


def main():
    student_path = 'data/student-mat.csv'

    test1_path = 'data/student-mat-test-1.csv'
    print_classification_report_from_csv(student_path, test1_path, 'wavfiles 1')

    test2_path = 'data/student-mat-test-2.csv'
    print_classification_report_from_csv(student_path, test2_path, 'wavfiles 2')

    test3_path = 'data/student-mat-test-3.csv'
    print_classification_report_from_csv(student_path, test3_path, 'wavfiles 3')

    test4_path = 'data/student-mat-test-4.csv'
    print_classification_report_from_csv(student_path, test4_path, 'wavfiles 4')

    test5_path = 'data/student-mat-test-5.csv'
    print_classification_report_from_csv(student_path, test5_path, 'wavfiles 5')


def print_classification_report_from_csv(student_path, test1_path, test_name):
    X_test, X_train, y_test, y_train = train_test_data_from_csv(test1_path, student_path)
    clf = StudentClassifier(algorithm)
    clf.train_classifier(X_train, y_train)
    predictions = clf.get_predictions(X_test)
    accuracy = clf.get_accuracy(X_test, y_test)
    print(f'Accuracy {test_name} =', accuracy)
    print(classification_report(y_test, predictions))


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
