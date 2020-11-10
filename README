# Predictor
Predictor is a wrapper for linear regression library `sklearn.linear_model`.
  
It's use is abstracted as much as possible from its original intent of predicting student's grades based on easy to know factors

Example:
```python
from predictor import Predictor

pred = Predictor(
    csv_file="student-mat.csv",
    data_columns=["G1", "G2", "G3", "failures", "absences", "famrel", "health", "higher", "internet"],
    prediction_column="G3",
    non_numerical_columns=["higher", "internet"],
    pickle_file="model.pickle"
)

pred.train(lambda avg, best: avg < 0.81)

pred.predict([20, 20, 0, 0, 5, 5, 1, 1])
```