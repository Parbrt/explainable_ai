import shap
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_prep import DataSet
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class SpecificDataSet(DataSet):
    def specific_prep(self, test_size=0.2, one_hot_enc=True):
        X_train, X_test, y_train, y_test = super().prep_data(test_size, one_hot_enc)
        self.feature_names = X_train.columns
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

def train_mlp(X_train, y_train):
    model = MLPClassifier(random_state=42, max_iter=1000, alpha=1.0, hidden_layer_sizes=(100,), learning_rate_init=0.01)
    return model.fit(X_train, y_train)

def get_shap_values(model, X_train, X_test):
    explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_train, 10))
    shap_vals = explainer.shap_values(X_test)
    
    # Extraction de la classe 1 (Good Credit) et de la base_value selon le format retourné
    sv_class1 = shap_vals[1] if isinstance(shap_vals, list) else (shap_vals[:, :, 1] if len(np.shape(shap_vals)) == 3 else shap_vals)
    expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    return sv_class1, expected_val

def plot_global(shap_values, X_data, feature_names):
    plt.figure()
    plt.title("Importance Globale (SHAP)")
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type="bar")

def plot_local(shap_values, expected_val, X_data, feature_names, index=0):
    exp = shap.Explanation(values=shap_values[index], base_values=expected_val, data=X_data[index], feature_names=feature_names)
    plt.figure()
    plt.title(f"Explicabilité Locale (Observation n°{index})")
    shap.waterfall_plot(exp)



# ============================================
#               Surrogate
# ============================================

def surrogate_dataset(X_train, model):
    surrogate_X = X_train.copy()
    surrogate_y = model.predict(X_train)
    return surrogate_X, surrogate_y

def surrogate_decisionTree_model(X_train, y_train, model, feature_names=None):
    surrogate_X, surrogate_y = surrogate_dataset(X_train, model)
    model_dt = DecisionTreeClassifier(random_state=42, max_depth=4)
    model_dt.fit(surrogate_X, surrogate_y)

    train_acc = model_dt.score(X_train, y_train)
    print("Decision Tree Surrogate AUC:", metrics.roc_auc_score(y_train, model_dt.predict(X_train)))

    plt.figure(figsize=(20, 10))
    plot_tree(model_dt, feature_names=feature_names, class_names=[str(c) for c in model_dt.classes_], filled=True)
    plt.title(f"Decision Tree | Train Acc: {train_acc:.4f} ", fontsize=14)
    plt.show()

def surrogate_linear_model(X_train, y_train, model):
    surrogate_X, surrogate_y = surrogate_dataset(X_train, model)
    model_lr = LinearRegression()
    model_lr.fit(surrogate_X, surrogate_y)

    train_acc = model_lr.score(X_train, y_train)
    print(f"Linear Regression Surrogate | Train Acc: {train_acc:.4f}")
    print("surrogate linear model AUC:", metrics.roc_auc_score(y_train, model_lr.predict(X_train)))




if __name__ == "__main__":
    # 1. Données
    dataset = SpecificDataSet()
    dataset.load_data()
    X_train, X_test, y_train, y_test = dataset.specific_prep()
    
    # 2. Entraînement & Évaluation
    model = train_mlp(X_train, y_train)
    print("--- Rapport de Classification ---\n", classification_report(y_test, model.predict(X_test)))
    print("Accuracy:", metrics.accuracy_score(y_test, model.predict(X_test)))
    print("MLP AUC:", metrics.roc_auc_score(y_test, model.predict(X_test)))
    
    # 3. Calcul SHAP
    print("Calcul des valeurs SHAP en cours...")
    shap_vals, expected_val = get_shap_values(model, X_train, X_test)
    
    # 4. Affichage
    plot_global(shap_vals, X_test, dataset.feature_names)
    
    random_idx = np.random.randint(0, len(X_test))
    plot_local(shap_vals, expected_val, X_test, dataset.feature_names, index=random_idx)
    plt.show()



    # Surrogate model
    surrogate_decisionTree_model(X_train, y_train, model, feature_names=list(dataset.feature_names))
    surrogate_linear_model(X_train, y_train, model)

