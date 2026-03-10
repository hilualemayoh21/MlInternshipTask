from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train):

    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train, y_train)

    return lr


def train_decision_tree(X_train, y_train):

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    return dt