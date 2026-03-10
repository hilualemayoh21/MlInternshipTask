import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def handle_missing_values(df):

    # Fix Dependents column (convert "3+" to 3)
    df['Dependents'] = df['Dependents'].replace('3+', 3)

    # convert to numeric
    df['Dependents'] = pd.to_numeric(df['Dependents'])

    categorical_cols = [
        'Gender','Married','Dependents',
        'Self_Employed','Credit_History'
    ]

    numerical_cols = [
        'LoanAmount','Loan_Amount_Term',
        'ApplicantIncome','CoapplicantIncome'
    ]

    # Fix categorical missing values
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Fix numeric missing values
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def encode_categorical(df):

    le = LabelEncoder()

    cat_cols = [
        'Gender','Married','Education',
        'Self_Employed','Property_Area'
    ]

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df