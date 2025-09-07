import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import json

X = None
y = None
scaler = None
model = None

def load_data(csv, drop, seed):
    global X, y, scaler, model
    data = pd.read_csv(csv)
    X = data.iloc[:, 2:6]
    if "TBI" in csv:
        #Without ALDOC
        #X = pd.concat([X, data.iloc[:, -2]], axis=1)
        #With ALDOC
        X = pd.concat([X, data.iloc[:, -2:]], axis=1)
    
    if (drop != "none"):
        X = X.drop(columns=[drop], errors="ignore")

    y = data.iloc[:, 0]
    y = y.apply(lambda label: 0 if label.lower() == 'control' else 1)
    ids = data.iloc[:, 0:2]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if "TBI" in csv:
        #Without ALDOC
        #mlp = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(32, 16), learning_rate='constant', solver='sgd', max_iter=1000, random_state=42)
        #With ALDOC
        mlp = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(32, 16), learning_rate='constant', solver='adam', max_iter=1000, random_state=42)
    elif "AD" in csv:
        mlp = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(32, 16), learning_rate='constant', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    model = mlp

    # y_pred = mlp.predict(X_test)
    # y_predprob = mlp.predict_proba(X_test)
    # pd.DataFrame(y_predprob, columns=["prob_control", "prob_tbi"]).to_csv("y_test_probs.csv", index=False)

    # # Predict on test set
    # y_pred = mlp.predict(X_test)
    # get_results(y_test, y_pred)

    # # Predict on train set
    # y_pred2 = mlp.predict(X_train)
    # get_results(y_train, y_pred2)

    # # Evaluate on the full dataset (train + test)
    # X_full = np.vstack([X_train, X_test])
    # y_full = pd.concat([y_train, y_test])
    # y_pred_full = mlp.predict(X_full)
    # get_results(y_full, y_pred_full)

    # y_pred_full = mlp.predict_proba(X_full)[:, 1]
    # pd.DataFrame(y_pred_full, columns=["prob_control", "prob_tbi"]).to_csv("y_pred_full_probs.csv", index=False)

def predict_new(gfap, ngrn, st2, bdnf, aldoc):
    global scaler, model
    if scaler is None or model is None:
        raise ValueError("Model and scaler must be loaded by calling load_data() before prediction.")

    new_data = np.array([[gfap, ngrn, st2, bdnf, aldoc]])
    # print(new_data)

    test_row = pd.DataFrame({
        "ADB/gfap": [gfap],
        "ADB/ngrn": [ngrn],
        "ADB/st2": [st2],
        "ADB/bdnf": [bdnf],
        "ALDOC_median": [aldoc]
    })

    new_data_scaled = scaler.transform(test_row)
    predictions = model.predict(new_data_scaled)
    print("prob_tbi:", model.predict_proba(new_data_scaled)[:, 1])

    return predictions

# Evaluate the model
def get_results(test, pred):
    cm = confusion_matrix(test, pred, labels=[0, 1])
    #print("Confusion Matrix:\n", cm)
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    accuracy = (tp + tn) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    ppa = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
    npa = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity

    #print("Overall Accuracy:", accuracy_score(test, pred))
    #print(classification_report(test, pred))
    print("Overall Accuracy: {:.2f}%".format(accuracy * 100))
    print("PPA (Sensitivity): {:.2f}%".format(ppa * 100))
    print("NPA (Specificity): {:.2f}%".format(npa * 100))
    print("Sum: {:.2f}%".format((ppa + npa) * 100))

load_data('new_parsed_TBI.csv', 'GCS', 42)

# Inputs: gfap, ngrn, st2, bdnf, aldoc. Measurement in ADB

print(predict_new(17.0, 1.0, 3087.5, 4153.5, 168773.75))  # false negative - TBI KH24-12368
print(predict_new(13.0, 109.0, 9885.0, 1888.0, 642.95))  # true positive - TBI KH25-00503
print(predict_new(8.5, 0.0, 6722.0, 954.5, 1097.75))  # true negative - Control HMN1243449A (012M)
print(predict_new(3.5, 0.0, 4476.0, 819.0, 3836.8))  # true negative - Control HMN1265412A (029F)

# if __name__ == "__main__":
#     # Read JSON input from stdin (from biomarkeranalysis.tsx)
#     input_data = sys.stdin.read()
#     try:
#         params = json.loads(input_data)
#         gfap = params.get("gfap")
#         ngrn = params.get("ngrn")
#         st2 = params.get("st2")
#         bdnf = params.get("bdnf")
#         aldoc = params.get("aldoc")
#         load_data('new_parsed_TBI.csv', 'GCS', 42)
#         result = predict_new(gfap, ngrn, st2, bdnf, aldoc)
#         print(json.dumps({"prediction": int(result[0])}))
#     except Exception as e:
#         print(json.dumps({"error": str(e)}))