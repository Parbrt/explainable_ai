import shap
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_prep import DataSet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score


class SpecificDataSet(DataSet):
    def specific_prep(self, test_size=0.2, one_hot_enc=True):
        X_train, X_test, y_train, y_test = super().prep_data(test_size, one_hot_enc)
        self.feature_names = X_train.columns
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=100)
    return model.fit(X_train, y_train)

def get_shap_values(model, X_train, X_test):
    explainer = shap.LinearExplainer(model, X_train)
    return explainer.shap_values(X_test), explainer.expected_value

def plot_global(shap_values, X_data, feature_names):
    plt.figure()
    plt.title("Importance Globale (Régression Logistique)")
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type="bar")

def plot_local(shap_values, expected_val, X_data, feature_names, index=0):
    exp = shap.Explanation(values=shap_values[index], base_values=expected_val, data=X_data[index], feature_names=feature_names)
    plt.figure()
    plt.title(f"Explicabilité Locale (Obs n°{index})")
    shap.waterfall_plot(exp)


if __name__ == "__main__":
    dataset = SpecificDataSet()
    dataset.load_data()
    X_train, X_test, y_train, y_test = dataset.specific_prep()
    
    model = train_logistic_regression(X_train, y_train)
    print("--- Rapport de Classification ---\n", classification_report(y_test, model.predict(X_test)))
    
    y_proba = model.predict_proba(X_test)[:, 1]
    print("AUC :", roc_auc_score(y_test, y_proba))
    
    shap_vals, expected_val = get_shap_values(model, X_train, X_test)
    
    plot_global(shap_vals, X_test, dataset.feature_names)
    
    random_idx = np.random.randint(0, len(X_test))
    plot_local(shap_vals, expected_val, X_test, dataset.feature_names, index=random_idx)
    
    plt.show()