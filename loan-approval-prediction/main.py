from src.data_processing import load_data, handle_missing_values, encode_categorical
from src.resample import balance_data
from src.models import train_logistic_regression, train_decision_tree
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.feature_engineering import scale_features

# load datasets
train_df, test_df = load_data("data/train.csv", "data/test.csv")

# preprocessing
train_df = handle_missing_values(train_df)
test_df = handle_missing_values(test_df)

train_df = encode_categorical(train_df)
test_df = encode_categorical(test_df)


# encode target variable
le = LabelEncoder()
train_df['Loan_Status'] = le.fit_transform(train_df['Loan_Status'])


# features and target
X = train_df.drop(['Loan_Status','Loan_ID'], axis=1)
y = train_df['Loan_Status']


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# balance data
X_train_res, y_train_res = balance_data(X_train, y_train)

# scale features
X_train_res, X_test = scale_features(X_train_res, X_test)

# train models
lr_model = train_logistic_regression(X_train_res, y_train_res)
dt_model = train_decision_tree(X_train_res, y_train_res)


# evaluation
print("Logistic Regression Performance")
evaluate_model(lr_model, X_test, y_test)

print("Decision Tree Performance")
evaluate_model(dt_model, X_test, y_test)