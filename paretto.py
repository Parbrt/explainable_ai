import matplotlib.pyplot as plt

models = ['BAYESIAN', 'MLP', 'XGBoost', 'Decision Tree', 'Logistic Regression', 'XGBoost_SHAP', 'MLP_SHAP']
accuracy = [0.75, 0.80, 0.805, 0.76, 0.81, 0.80, 0.805]
AUC = [0.66, 0.80, 0.79, 0.76, 0.81, 0.80, 0.79]
explainability = [7.5, 1, 2, 8.5, 7, 6, 6]

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