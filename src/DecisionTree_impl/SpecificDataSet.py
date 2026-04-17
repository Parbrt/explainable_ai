"""
    X-AI Project
    ESAIP 04/2026
    HERBRETEAU Mathis, LE POTTIER Mathias, ROBERT Paul-Aimé
"""
from src.utils.data_prep import DataSet

class SpecificDataSet(DataSet):
    def __init__(self):
        super().__init__()
    def specific_prep(self):
        if self.df is not None:
            for col in self.categorical_columns:
                self.df[col] = self.df[col].astype("category")
        self.X_train, self.X_test, self.y_test, self.y_train = self.prep_data(one_hot_enc=True)
        return self.X_train, self.X_test, self.y_test, self.y_train

if __name__ == "__main__":
    dataset = SpecificDataSet()
    dataset.specific_prep()
