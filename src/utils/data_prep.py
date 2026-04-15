"""
    X-AI Project
    ESAIP 04/2026
    HERBRETEAU Mathis, LE POTTIER Mathias, ROBERT Paul-Aimé
"""

import pandas as pd
from sklearn.model_selection import train_test_split

PATH = 'data/german_credit_data.csv'

NUM_COLS = [
    'month_duration',
    'credit_amount',
    'payment_to_income_ratio',
    'residence_since',
    'age',
    'n_credits',
    'n_guarantors'
]

CAT_COLS = [
    'status_account',
    'credit_history',
    'purpose',
    'status_savings',
    'years_employment',
    'status_and_sex',
    'secondary_obligor',
    'collateral',
    'other_installment_plans',
    'housing',
    'job',
    'telephone',
    'is_foreign_worker'
]


class DataSet():
    """
        Class Data set
    """
    def __init__(self,data_path = PATH,categorical_columns = CAT_COLS,numeric_columns = NUM_COLS):
        """
            input: str
        """
        self.data_path = data_path
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.df = self.load_data()

    def load_data(self):
        """
            Load data from csv
            input:
                data_path: String
            output:
                df: Pandas DataFrame
        """
        try:
            self.df = pd.read_csv(self.data_path, encoding='ascii', delimiter=',')
            print('Data loaded successfully.')
        except Exception as e:
            print('Error loading data:', e)
            self.df = None

        return self.df

    def prep_data(self, test_size = 0.2,one_hot_enc = True) :
        """
            Prepare data
            input:
                test_size: Float
                one_hot_enc: Bool: Are we changing categorical columns to one hot encoding ?
            output:
                X_train, X_test, y_train, y_test: Pandas DataFrame

        """
        if self.df is None :
            print ('Error loading data: df passed as an agrgument is None')
            return None

        for col in self.numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df['target_flag'] = self.df['target'].apply(lambda x: 1 if x.strip().lower() == 'good' else 0)

        features = self.df.drop(columns=['target', 'target_flag'])
        target = self.df['target_flag']

        if one_hot_enc:
            features_encoded = pd.get_dummies(features, drop_first=True)
            features_encoded = features_encoded.fillna(features_encoded.mean())
        else :
            features_encoded = features

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features_encoded, target, test_size=test_size, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
