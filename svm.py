from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, stratify=y, random_state=42
)

svm_clf =make_pipeline(StandardScaler(),
                           LinearSVC(random_state=42))
svm_score = svm_clf.fit(X_train, y_train).score(X_test, y_test)

print('svm score', svm_score*100)