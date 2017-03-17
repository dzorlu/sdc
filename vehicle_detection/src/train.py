from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pickle

MODEL_PATH = "saved_models/"

def save_model(model_fit, model_name):
    # Persist model for lookup at test time
    _file = open(MODEL_PATH + model_name ,'wb')
    pickle.dump(model_fit, _file)
    _file.close()
    print("Model {} saved..".format(model_name))

def train_model(X,y):
    t=time.time()

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    save_model(scaler,"standard_scaler")
    X = scaler.transform(X)

    # SVM Fit
    svc = LinearSVC(penalty='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Save the model
    save_model(svc,"linear_svm")
