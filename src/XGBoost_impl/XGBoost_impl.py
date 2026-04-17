"""
    X-AI Project
    ESAIP 04/2026
    HERBRETEAU Mathis, LE POTTIER Mathias, ROBERT Paul-Aimé
"""

import shap
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import numpy as np
from src.XGBoost_impl.SpecificDataSet import SpecificDataSet
import matplotlib.pyplot as plt


class XGBoostImpl():
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.data_set = SpecificDataSet()
        self.X_train,self.X_test,self.y_train, self.y_test = self.data_set.specific_prep()


    def train(self):
        """
        training
        input:
            test_size (float): size of the testing sample
        output:
            model
        """
        print("Début de l'entraînement...")
        
        
        self.model = xgb.XGBClassifier(enable_categorical = True)

        self.model.fit(self.X_train, self.y_train)
        print("Entraînement terminé.")

        self.evaluate()
        return self.model

    def evaluate(self):
        """
        evaluate the model trained
        """
        print("=====Eval=====")

        if self.model is None:
            raise Exception("Le modèle n'est pas entraîné.")
        # y_pred -> numpy array (n_samples, n_targets)
        y_pred = self.model.predict(self.X_test)
        
        # Récupération des vraies valeurs et des prédictions pour cette colonne
        true_vals = self.y_test.values
        pred_vals = y_pred

        report = classification_report(true_vals, pred_vals, output_dict=True, zero_division=0)
            
        f1 = report['weighted avg']['f1-score']
        acc = report['accuracy']
            
        print(f"* Accuracy : {acc:.2%}")
        print(f"* F1-Score : {f1:.2%}")
            
    def predict(self, input_data):
        """
         
        """
        if self.model is None:
            raise Exception("Le modèle n'est pas entraîné.")
            
        
        input_scaled = self.scaler.transform(input_data)
        return self.model.predict(input_scaled)


    def test_random_prediction(self, do_explain = False):
        """
        Sélectionne une ligne aléatoire dans le set de test, 
        prédit les cibles et affiche la comparaison avec le réel.
        """
        if self.model is None:
            print("Erreur : Le modèle doit être entraîné avant le test.")
            return

        # 1. Sélectionner un index au hasard
        random_index = np.random.randint(0, len(self.X_test))
        
        # 2. Extraire la ligne (X_test est déjà un array numpy ici)
        # On utilise [random_index:random_index+1] pour garder la forme (1, n_features)
        sample_x = self.X_test[random_index:random_index+1]
        
        # 3. Extraire les vraies valeurs (y_test est un DataFrame)
        true_y = self.y_test.iloc[random_index]
        
        # 4. Faire la prédiction
        pred_y = self.model.predict(sample_x)[0] # On prend le premier (et seul) résultat

        print(f"=====Test sur ligne aléatoire (Index: {random_index})=====")

        print("columns :\n",sample_x)
        print("val réel : ",true_y)
        print("val prédite : ",pred_y)

        if do_explain:
            self.explain_line(random_index)

    def explain_line(self,line = 3):

        if self.model is None:
            print("Erreur : Le modèle doit être entraîné avant le test.")
            return
        # Create a tree explainer
        xgb_explainer = shap.TreeExplainer(self.model)


        shap_explainer_values = xgb_explainer(self.X_test)
        shap.waterfall_plot(shap_explainer_values[line])

        plt.show()

    def explain_global(self):

        if self.model is None:
            print("Erreur : Le modèle doit être entraîné avant le test.")
            return


        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_test)

        # Graphique global complet
        shap.summary_plot(shap_values, self.X_test)

        plt.show()
if __name__ == "__main__":
    model = XGBoostImpl()
    model.train()
    model.explain_global()
    input_word = 0
    while input_word != 1:
        input_word = input("press 1 to exit\n>>")
        model.test_random_prediction(do_explain=True)

