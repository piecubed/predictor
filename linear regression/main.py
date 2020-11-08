import pickle
import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection
import matplotlib.pyplot as pyplot
from matplotlib import style

data_columns = ["G1", "G3", "failures", "absences", "Medu", "famrel", "health"]
predict = "G3"


class StudentGradePredictor:
    __raw = None
    linear: linear_model = None
    __data = None
    __X = None
    __Y = None

    def __init__(self):
        try:
            self.__raw = pickle.load(open("linear regression/student_model.pickle", "rb"))
            self.linear = self.__raw["model"]
        except FileNotFoundError:
            self.linear = linear_model()
            __raw = {"best": 0, "model": self.linear}

        assert self.linear is not None
        assert self.__raw is not None

    def read_data(self):
        if self.__X is None or self.__Y is None or self.__data is None:
            self.__data = pd.read_csv("linear regression/student-mat.csv", sep=";")[data_columns]

            self.__X = np.array(self.__data.drop([predict], 1))
            self.__Y = np.array(self.__data[predict])

    def __pick_sample_data(self, test_size=0.1):
        self.read_data()

        return model_selection.train_test_split(self.__X, self.__Y, test_size=test_size)

    def train(self, condition):
        avg = 0
        c = 0
        best = self.__data["best"]
        while condition:
            c += 1
            # Pick sample data
            x_train, x_test, y_train, y_test = self.__pick_sample_data()

            # Train model
            self.linear.fit(x_train, y_train)

            # Test the model
            acc = self.linear.score(x_test, y_test)

            avg += acc
            print(f"{c}: Accuracy: {round(acc, 5)}    {c} Average: {round(avg / c, 5)}    {c}: Best: {round(best, 5)}")

            # Save the model if it performs better
            if acc > best:
                best = acc
                with open("student_model.pickle", "wb") as f:
                    pickle.dump({"best": best, "model": self.linear}, f)

    def predict(self, grade, failures, absences, medu, famrel, health):
        x = np.array([[
            grade,
            failures,
            absences,
            medu,
            famrel,
            health,
            # I have no idea what these do, but they are needed for the prediction to work.
            # TODO find out what these do
            0,
            0
        ]])
        y = self.linear.predict(x)[0]
        return y

    def graph(self, p):
        self.read_data()
        style.use('ggplot')
        pyplot.scatter(self.__data[p], self.__data['G3'])
        pyplot.xlabel(p)
        pyplot.ylabel("Final Grade")
        pyplot.show()

    def ask_for_prediction(self):
        gpa = float(input("Grade (Out of 20): "))
        failures = int(input("Failures: "))
        absences = int(input("Absences: "))
        medu = int(
            input("Mothers Education (0: none, 1: 1st-5th grade, 2: 5th-9th grade, 3: 9th-12th, 4: Higher education) "))
        famrel = int(input("On a scale of 1, very poor, to 5, excellent, what is your family relationship?  "))
        health = int(input("On a scale of 1, very poor, to 5, very good, what is your health? "))

        print("Your final grade:", self.predict(gpa, failures, absences, medu, famrel, health))
