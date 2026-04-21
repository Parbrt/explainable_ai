import matplotlib.pyplot as plt

models = ['BAYESIAN', 'Surro decisionTree', 'Surro linear', 'MLP', 'XGBoost', 'Decision Tree', 'Logistic Regression']
accuracy = [0.75, 0.75, 0.28, 0.80, 0.805, 0.76, 0.81]
AUC = [0.66, 0.67, 0.83, 0.74, 0.73, 0.66, 0.74]
explainability = [7.5, 8.5, 8, 1, 2, 8.5, 7]

points = list(zip(models, AUC, explainability))
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


plt.scatter(explainability, AUC, color='royalblue', s=100, label='Modèles (Dominés)')

plt.plot(p_exp, p_acc, color='crimson', marker='o', markersize=10, linestyle='--', linewidth=2, label='Front de Pareto')


for i, model in enumerate(models):
    plt.annotate(model,
                 (explainability[i], AUC[i]),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center',
                 fontsize=9)


plt.title("Comparaison de modèles : Explicabilité vs AUC")
plt.xlabel("Explicabilité (Score)")
plt.ylabel("AUC")
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower left')
plt.show()