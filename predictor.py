import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import linear_model, model_selection
from sklearn.preprocessing import LabelEncoder


class Predictor:
    __csv_file: str = None
    __csv_sep: str = None
    __data_columns: List[str] = None
    __df: pd.DataFrame = None
    __non_numerical_columns: List[str] = []
    __pickle_file: str = None
    __prediction_column: str = None
    __raw: Dict[str, Any] = None
    __X: np.array = None
    __Y: np.array = None

    linear: linear_model = None

    def __init__(self, *, csv_file: str, data_columns: List[str], prediction_column: str,
                 pickle_file: Optional[str], csv_sep: str = ",",
                 non_numerical_columns: Optional[List[str]] = None) -> None:

        """
        Predicts a data column based on other columns using Linear Regression.

        :param csv_file: str What file the CSV data is in.
        :param data_columns: List[str] What columns of data to include from the CSV file.
        :param prediction_column: str What column to predict.
        :param pickle_file: Optional[str] What file the pickled model is in.
        :param non_numerical_columns: Optional[List[str]] What columns need to be converted to numerical values.
        :param csv_sep: str What separator the CSV file has.
        """

        if non_numerical_columns is not None:
            self.__non_numerical_columns = non_numerical_columns

        self.__csv_file = csv_file
        self.__csv_sep = csv_sep
        self.__data_columns = data_columns
        self.__non_numerical_columns = non_numerical_columns
        self.__pickle_file = pickle_file
        self.__prediction_column = prediction_column

        try:
            if self.__pickle_file is None:
                raise FileNotFoundError()

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
            print("WARNING: Model cache based on old columns, creating new model.")
            self.linear = linear_model.LinearRegression()

            self.__raw = {"best": 0, "model": self.linear, "columns": data_columns}
            self.__save_data()

        assert self.linear is not None
        assert self.__raw is not None

        self.__read_data()

    def __save_data(self) -> None:
        """
        Saves __data to __pickle_file.
        """
        with open(self.__pickle_file, "wb") as f:
            pickle.dump(self.__raw, f)

    def __transform_non_numerical_column(self, column: str) -> None:
        """
        Transforms a non-numerical column into
        :param column: str Name of the column to transform.
        """
        # Try to convert a string datatype to a numerical value
        enc = LabelEncoder()
        enc.fit(self.__df[column])
        self.__df[column] = enc.transform(self.__df[column])

    def __pick_sample_data(self, test_size: int = 0.1) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Picks sample data

        :param test_size:
        :return: (x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array)
        """
        return model_selection.train_test_split(self.__X, self.__Y, test_size=test_size)

    def __read_data(self) -> None:
        """
        Reads and sanitizes data.  Safe to be called multiple times.
        :return: None
        """
        if self.__X is None or self.__Y is None or self.__df is None:
            # Read CSV file
            self.__df = pd.read_csv(self.__csv_file, sep=self.__csv_sep)[self.__data_columns]

            # Transform any specified non numerical columns
            for column in self.__non_numerical_columns:
                self.__transform_non_numerical_column(column)

            # Determine X and Y
            self.__X = np.array(self.__df.drop([self.__prediction_column], 1))
            self.__Y = np.array(self.__df[self.__prediction_column])

    def train(self, condition: Callable[[float, float], bool], min_fits: int = 10, silent: bool = False,
              save_model: bool = True) -> None:
        """
        Trains the model until `condition` is met.
        :param condition: Callable[[float, float], bool] A function that decides when to stop training.
        Takes two arguments, average accuracy, and best accuracy.
        :param min_fits: int Min number of fits to do.
        :param silent: bool Controls if accuracy is printed.
        :param save_model: bool Controls if the model is pickled after `condition` is met.
        :return: None
        """
        total = 0
        c = 0
        best = self.__raw["best"]
        while condition(0 if c == 0 else total / c, best) or c < min_fits:
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

    def predict(self, data: List[Any]) -> float:
        """
        Attempt to predict a value.
        :param data: List[Any] Input to predict a value with.
        :return: Predicted value.
        """
        x = np.array([data])
        y = self.linear.predict(x)[0]
        return y

    def graph(self, column: str, column_label: str = None) -> None:
        """
        Graph a column
        :param column: str Name of the column to show
        :param column_label: str Label of the column
        :return: None
        """
        style.use('ggplot')
        pyplot.scatter(self.__df[column], self.__df[self.__prediction_column])
        pyplot.xlabel(column_label if column_label is not None else column)
        pyplot.ylabel(self.__prediction_column)
        pyplot.show()


def ask_for_prediction(predictor: Predictor) -> None:
    """
    Gets user input and predicts a grade
    :param predictor: Predictor Instance of Predictor
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
    print("Your final grade:", predictor.predict([G1, G2, failures, absences, famrel, health, higher, internet]))


if __name__ == "__main__":
    __pred = Predictor(
        csv_file="student-mat.csv",
        csv_sep=";",
        data_columns=["G1", "G2", "G3", "failures", "absences", "famrel", "health", "higher", "internet"],
        prediction_column="G3",
        non_numerical_columns=["higher", "internet"],
        pickle_file="model.pickle"
    )

    __pred.train(lambda avg, best: avg < 0.80)

    ask_for_prediction(__pred)
