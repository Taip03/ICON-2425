from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True)),
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "NaiveBayes": GaussianNB()
}
