import sys
from pathlib import Path
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, classification_report
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

FILE_DIR = Path(__file__).resolve().parent
SRC_DIR = FILE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.data_prep import DataSet

PATH = str(PROJECT_ROOT / 'data' / 'german_credit_data.csv')

# Discretisation des colonnes continues pour le réseau bayésien
DISCRETIZE_COLS = {
    'month_duration': {
        'bins': [0, 12, 24, 36, 72],
        'labels': ['court', 'moyen', 'long', 'tres_long']
    },
    'credit_amount': {
        'bins': [0, 2000, 5000, 10000, 20000],
        'labels': ['faible', 'moyen', 'eleve', 'tres_eleve']
    },
    'age': {
        'bins': [0, 25, 40, 60, 100],
        'labels': ['jeune', 'adulte', 'senior', 'aine']
    }
}

class SpecificDataSet(DataSet):
    def __init__(self, data_path=PATH):
        super().__init__(data_path)

    def discretize(self, df):
        """Discrétise les colonnes continues pour le réseau bayésien."""
        df = df.copy()
        for col, params in DISCRETIZE_COLS.items():
            if col in df.columns:
                df[col] = pd.cut(
                    df[col],
                    bins=params['bins'],
                    labels=params['labels'],
                    include_lowest=True
                ).astype(str)
        return df

    def specific_prep(self):
        # Pas de one-hot encoding : les variables catégorielles restent telles quelles
        self.X_train, self.X_test, self.y_train, self.y_test = self.prep_data(one_hot_enc=False)
        self.X_train = self.discretize(self.X_train)
        self.X_test  = self.discretize(self.X_test)
        self.X_train = self.X_train.astype(str)
        self.X_test  = self.X_test.astype(str)

data = SpecificDataSet()
data.specific_prep()

target = 'target_flag'

train_df = data.X_train.copy()
train_df[target] = data.y_train.astype(str)


#==========================================
#   Graphe bayésien : structure d'apprentissage
#==========================================


def learn_bayesian_structure(train_df):
    hc = HillClimbSearch(train_df)
    best_model_structure = hc.estimate(scoring_method="bic-d")
    print("Edges of the learned network:", best_model_structure.edges())

    learned_model = DiscreteBayesianNetwork(best_model_structure.edges())
    plt.figure(figsize=(8, 6))
    pos = nx.circular_layout(learned_model)
    nx.draw(learned_model, pos=pos, with_labels=True, node_color='lightblue', node_size=2500, font_size=10, arrowsize=20)
    plt.title("Learned Bayesian Network Structure")
    output_path = FILE_DIR / "bayesian_network.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphe sauvegardé : {output_path}")

    return learned_model


#==========================================
#   Table de probabilités conditionnelles
#==========================================

def learn_bayesian_cpds(learned_model, train_df):

    learned_model.fit(train_df, estimator=MaximumLikelihoodEstimator)

    print("="*32)
    print("Conditional Probability Table for 'target_flag':")
    print(learned_model.get_cpds('target_flag'))
    print("="*32)

    print("Automated bayesian structure")
    df_cpd = learned_model.get_cpds('target_flag').to_dataframe()
    print("="*32)
    print(df_cpd)
    print("="*32)




#==========================================
#           Variable Elimination
#==========================================

def variable_elimination_inference(learned_model):
    inference = VariableElimination(learned_model)
    node_labels_with_probs = {}
    for node in learned_model.nodes():
        marginal_dist = inference.query(variables=[node], show_progress=False)

        label_str = f"{node}\n"
        states = marginal_dist.state_names[node]
        probabilities = marginal_dist.values

        for state, prob in zip(states, probabilities):
            label_str += f"{state}: {prob:.2f}\n"
        node_labels_with_probs[node] = label_str.strip()

    plt.figure(figsize=(12, 8))
    pos = nx.circular_layout(learned_model) 
    nx.draw(
        learned_model,
        pos=pos,
        with_labels=True,
        labels=node_labels_with_probs, 
        node_color='lightblue',
        node_size=5000, 
        font_size=8,
        font_weight='bold',
        arrowsize=20,
        edge_color='gray'
    )
    plt.title("Learned Bayesian Network Structure with Marginal Probabilities", size=15)
    plt.axis('off') 
    output_path = FILE_DIR / "bayesian_network_VariableElimination.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphe sauvegardé : {output_path}")

    return inference


#==========================================
#           Calcul accuracy
#==========================================


def predict_bayesian(model, X_test, target, inference_engine):
    predictions = []
    for _, row in X_test.iterrows():
        evidence = {col: row[col] for col in X_test.columns if col in model.nodes()}
        try:
            result = inference_engine.map_query(
                variables=[target],
                evidence=evidence,
                show_progress=False
            )
            predictions.append(result[target])
        except Exception:
            predictions.append(None)
    return predictions

def calculate_accuracy(model, inferenceModel) :
    test_df = data.X_test.copy()
    test_df = test_df.astype(str)

    y_pred = predict_bayesian(model, test_df, target, inferenceModel)
    y_true = data.y_test.astype(str).tolist()

    valid = [(p, t) for p, t in zip(y_pred, y_true) if p is not None]
    y_pred_clean, y_true_clean = zip(*valid)

    print("="*32)
    print(f"Accuracy: {accuracy_score(y_true_clean, y_pred_clean):.4f}")
    print(classification_report(y_true_clean, y_pred_clean))
    print("="*32)




#==========================================
#           Calcul accuracy
#==========================================

def inference_example(inference_engine, evidence):
    result = inference_engine.query(variables=['target_flag'], evidence=evidence, show_progress=False)
    print("="*32)
    print("Inference result for evidence:", evidence)
    print(result) 
    print("="*32)


if __name__ == "__main__":

    evidence = {
        'month_duration': 'court',
        'credit_amount': 'faible',
        'age': 'jeune'
    }


    learned_model = learn_bayesian_structure(train_df)
    learn_bayesian_cpds(learned_model, train_df)
    inference_engine = variable_elimination_inference(learned_model)
    calculate_accuracy(learned_model, inference_engine)
    inference_example(inference_engine, evidence)