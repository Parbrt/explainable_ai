import matplotlib.pyplot as plt

models = ['BAYESIAN', 'Surro decisionTree', 'Surro linear', 'MLP', 'XGBoost']
accuracy = [0.75, 0.75, 0.28, 0.80, 0.85]
explainability = [7.5, 8.5, 9, 1, 3.5]

points = list(zip(models, accuracy, explainability))
points_sorted = sorted(points, key=lambda x: (x[1], x[2]), reverse=True)

pareto_front = []
max_explainability = -1

for p in points_sorted:
    if p[2] > max_explainability:
        pareto_front.append(p)
        max_explainability = p[2]


pareto_front_sorted = sorted(pareto_front, key=lambda x: x[1])
p_models, p_acc, p_exp = zip(*pareto_front_sorted)


plt.figure(figsize=(10, 6))


plt.scatter(accuracy, explainability, color='royalblue', s=100, label='Modèles (Dominés)')

plt.plot(p_acc, p_exp, color='crimson', marker='o', markersize=10, linestyle='--', linewidth=2, label='Front de Pareto')


for i, model in enumerate(models):
    plt.annotate(model, 
                 (accuracy[i], explainability[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', 
                 fontsize=9)


plt.title("Comparaison de modèles : Accuracy vs Explicabilité")
plt.xlabel("Accuracy")
plt.ylabel("Explicabilité (Score)")
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower left')
plt.show()