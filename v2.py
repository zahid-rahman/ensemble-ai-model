# stacking

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

# combine models
estimators = [
    ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42))),
    ('mlp', make_pipeline(StandardScaler(), MLPClassifier(alpha=1, max_iter=100))),
 ]
# meta train
clf = StackingClassifier(
     estimators=estimators, final_estimator=LogisticRegression(random_state=42)
 )

X_train, X_test, y_train, y_test = train_test_split(
     X, y, stratify=y, random_state=42
)

# train the combine model
prediction_score = clf.fit(X_train, y_train).score(X_test, y_test)
print('Ensemble accuracy:', prediction_score)
