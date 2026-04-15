from src.utils.data_prep import DataSet

class SpecificDataSet(DataSet):
    def __init__(self, data_path=..., categorical_columns=..., numeric_columns=...):
        super().__init__(data_path, categorical_columns, numeric_columns)
    def specific_prep(self):
        pass

dataset = SpecificDataSet()
dataset.prep_data()
dataset.specific_prep()
