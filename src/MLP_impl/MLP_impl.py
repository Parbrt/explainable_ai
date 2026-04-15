from src.utils.data_prep import DataSet
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

class SpecificDataSet(DataSet):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def specific_prep(self, test_size=0.2, one_hot_enc=True):
        """
        Préparation spécifique pour le MLP : exécute la préparation de base 
        puis applique la standardisation.
        """
        X_train, X_test, y_train, y_test = super().prep_data(test_size, one_hot_enc)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        
        return self.X_train, self.X_test, self.y_train, self.y_test


dataset = SpecificDataSet()
dataset.load_data()

X_train, X_test, y_train, y_test = dataset.specific_prep()


model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50))
model.fit(X_train, y_train)

mlp_predictions = model.predict(X_test)

print("--- MLP Classifier Results (Scaled Data) ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, mlp_predictions))
print("\nClassification Report:")
print(classification_report(y_test, mlp_predictions))