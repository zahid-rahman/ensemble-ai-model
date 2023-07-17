#basic technique
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

# Load your dataset and split it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual models
svm_model = LinearSVC(random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500)
rbf_model = SVC(kernel='rbf')
rt_model = RandomForestClassifier()

# Train individual models
svm_model.fit(X_train, y_train)
mlp_model.fit(X_train, y_train)
rbf_model.fit(X_train, y_train)
rt_model.fit(X_train, y_train)

# Make predictions using individual models
svm_preds = svm_model.predict(X_test)
mlp_preds = mlp_model.predict(X_test)
rbf_preds = rbf_model.predict(X_test)
rt_preds = rt_model.predict(X_test)

# Ensemble predictions using voting
ensemble_preds = (svm_preds + mlp_preds + rbf_preds + rt_preds) // 4

# Evaluate ensemble model accuracy
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
print("Ensemble Accuracy:", ensemble_accuracy * 100)






