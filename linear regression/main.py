import pickle

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import uuid

from matplotlib import style
from sklearn import linear_model, model_selection
from sklearn.preprocessing import LabelEncoder


class StudentGradePredictor:
    __pickle_file = None
    __raw = None
    linear: linear_model = None
    __df = None
    __X = None
    __Y = None
    __data_columns = None
    __prediction_column = None
    __non_numerical_columns = None

    def __init__(self, *, data_columns, prediction_column, non_numerical_columns,
                 pickle_file="linear regression/student_model.pickle"):
        self.__data_columns = data_columns
        self.__prediction_column = prediction_column
        self.__non_numerical_columns = non_numerical_columns

        self.__pickle_file = pickle_file

        try:
            self.__raw = pickle.load(open(self.__pickle_file, "rb"))
            if "best" not in self.__raw.keys() or "model" not in self.__raw.keys():
                raise KeyError()
            self.linear = self.__raw["model"]

        except (FileNotFoundError, KeyError):
            print("WARNING: Creating new model")
            self.linear = linear_model.LinearRegression()
            self.__raw = {"best": 0, "model": self.linear, "columns": data_columns}
            self.__save_data()

        if set(self.__raw["columns"]) != set(data_columns):
            print("Model cache based on old columns, updating pickle.")
            self.linear = linear_model.LinearRegression()

            self.__raw = {"best": 0, "model": self.linear, "columns": data_columns}
            self.__save_data()

        assert self.linear is not None
        assert self.__raw is not None

    def __save_data(self):
        with open(self.__pickle_file, "wb") as f:
            pickle.dump(self.__raw, f)

    def __transform_non_numerical_frame(self, frame):
        # Try to convert a string datatype to numerical
        enc = LabelEncoder()
        enc.fit(self.__df[frame])
        self.__df[frame] = enc.transform(self.__df[frame])

    def __pick_sample_data(self, test_size=0.1):
        self.read_data()

        return model_selection.train_test_split(self.__X, self.__Y, test_size=test_size)

    def read_data(self, csv_file="linear regression/student-mat.csv"):
        """
        Reads and sanitizes data.
        Safe to be called multiple times.
        :param csv_file: Default="linear regression/student-mat.csv".  Path to the CSV data file.
        :return: None
        """

        if self.__X is None or self.__Y is None or self.__df is None:
            # Read CSV file
            self.__df = pd.read_csv(csv_file, sep=";")[self.__data_columns]

            # Transform any specified non numerical frames
            for frame in self.__non_numerical_columns:
                self.__transform_non_numerical_frame(frame)

            # Determine X and Y
            self.__X = np.array(self.__df.drop([self.__prediction_column], 1))
            self.__Y = np.array(self.__df[self.__prediction_column])

    def train(self, condition, min_fits=10, silent=False, save_model=True):
        """
        Trains the model until `condition` is met.
        :param condition: A function that decides when to stop training.  Should take two arguments, average accuracy,
        and best accuracy.
        :param min_fits: Default=10 Min number of fits to do.
        :param silent: Default=False, controls if accuracy is printed.
        :param save_model: Default=True, controls if the model is pickled after `condition` is met.
        :return: None
        """

        self.read_data()

        total = 0
        c = 0
        best = self.__raw["best"]
        while condition(0 if c == 0 else total/c, best) or c < min_fits:
            c += 1
            # Pick sample data
            x_train, x_test, y_train, y_test = self.__pick_sample_data()

            # Train model
            self.linear.fit(x_train, y_train)

            # Test the model
            acc = self.linear.score(x_test, y_test)

            total += acc

            if not silent:
                print(f"{c}: Accuracy: {round(acc, 5)} {c} Average: {round(total / c, 5)} {c}: Best: {round(best, 5)}")

            # Save the model if it performs better
            if acc > best:
                best = acc

        if save_model:
            self.__raw['model'] = self.linear
            self.__save_data()

    def predict(self, data):
        """
        Attempt to predict a value.
        :param data: Input to predict a value with.
        :return: Predicted value.
        """

        x = np.array([data])
        y = self.linear.predict(x)[0]
        return y

    def graph(self, column, column_label=None):
        """
        Graph a frame
        :param column: Name of the column to show
        :param column_label: Label of the column
        :return: None
        """
        self.read_data()
        style.use('ggplot')
        pyplot.scatter(self.__df[column], self.__df[self.__prediction_column])
        pyplot.xlabel(column_label if column_label is not None else column)
        pyplot.ylabel("Final Grade")
        pyplot.show()

    # NOT GENERALIZED
    def ask_for_prediction(self):
        """
        Gets user input and predicts a grade
        :return: None
        """
        G1 = float(input("Grade first quarter (Out of 20): "))
        G2 = float(input("Grade second quarter (Out of 20): "))
        failures = int(input("Failures: "))
        absences = int(input("Absences: "))
        famrel = int(input("On a scale of 1, very poor, to 5, excellent, what is your family relationship?  "))
        health = int(input("On a scale of 1, very poor, to 5, very good, what is your health? "))
        higher = int(input("Do you want to pursue a higher education? (y, n) ") == "y")
        internet = int(input("Do you have internet access at home? (y, n) ") == "y")
        print("Your final grade:", self.predict([G1, G2, failures, absences, famrel, health, higher, internet]))


# Instead of putting the pickle file into version control, we decided to use the node ID of our computers to keep our
# models separate.
# If you are using the other pickle file, be sure to remember to turn on no_save
def get_pickle_file():
    return f'linear regression/{uuid.getnode()}_model.pickle'


if __name__ == "__main__":
    sgp = StudentGradePredictor(
        data_columns=["G1", "G2", "G3", "failures", "absences", "famrel", "health", "higher", "internet"],
        prediction_column="G3",
        non_numerical_columns=["higher", "internet"],
        pickle_file=get_pickle_file()
    )

    sgp.train(lambda avg, best: avg < 0.80)

    sgp.ask_for_prediction()
