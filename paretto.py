import matplotlib.pyplot as plt
# 1. Données
models = ['BAYESIAN', 'Surro decisionTree', 'Surro linear', 'MLP', 'XGBoost']
accuracy = [0.75, 0.75, 0.28, 0.80, 0.85]
explainability = [7.5, 8.5, 9, 1, 3.5]

points = list(zip(models, accuracy, explainability))

# 2. Calcul du Front de Pareto (Maximiser l'Accuracy ET l'Explicabilité)
# On trie d'abord les modèles par Accuracy décroissante
points_sorted = sorted(points, key=lambda x: x[1], reverse=True)

pareto_front = []
max_explainability = -1

# On parcourt les modèles. On ne garde un modèle que si son explicabilité 
# est strictement supérieure à la meilleure explicabilité rencontrée jusqu'ici.
for p in points_sorted:
    if p[2] > max_explainability:
        pareto_front.append(p)
        max_explainability = p[2]

# On trie le front de Pareto par Accuracy croissante pour que la ligne se trace correctement de gauche à droite
pareto_front_sorted = sorted(pareto_front, key=lambda x: x[1])
p_models, p_acc, p_exp = zip(*pareto_front_sorted)

# 3. Tracé du graphique
plt.figure(figsize=(10, 6))

# Tracer tous les modèles en bleu
plt.scatter(accuracy, explainability, color='royalblue', s=100, label='Modèles (Dominés)')

# Tracer la ligne et les points du front de Pareto en rouge
plt.plot(p_acc, p_exp, color='crimson', marker='o', markersize=10, linestyle='--', linewidth=2, label='Front de Pareto')

# Ajouter les étiquettes de texte pour chaque modèle
for i, model in enumerate(models):
    plt.annotate(model, 
                 (accuracy[i], explainability[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', 
                 fontsize=9)

# Configuration esthétique du graphe
plt.title("Comparaison de modèles : Accuracy vs Explicabilité")
plt.xlabel("Accuracy")
plt.ylabel("Explicabilité (Score)")
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower left')

# Afficher le graphique
plt.show()