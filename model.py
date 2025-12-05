from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self, x_train, y_train):
        self.model = LogisticRegression(max_iter = 1000)
        self.model.fit(x_train, y_train)

    def pred(self, test):
        self.predictions = self.model.predict(test)
        return self.predictions
    
    def model_accuracy(self, y_test, x_test):
        y_pred = self.model.predict(x_test)
        self.acc = accuracy_score(y_pred, y_test)
