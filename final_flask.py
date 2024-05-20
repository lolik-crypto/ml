import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from flask import Flask, render_template, url_for, request, jsonify

app = Flask(__name__)

diagnoz_df = {
    0: "высокая",
    1: "низкая"
}

menu = [{"name": "KNN", "url": "model_KNN"},
        {"name": "Logistic", "url": "model_Logistic"},
        {"name": "Wood", "url": "model_Wood"}]

loaded_model_knn = pickle.load(open('model_KNN/model_KNN', 'rb'))
loaded_model_Log = pickle.load(open('model_logistic/model_logistic', 'rb'))
loaded_model_Tree =pickle.load(open('model_wood/model_wood', 'rb'))


def classification_model_metrics(model: str) -> dict:
    models = {"knn": loaded_model_knn, "logistic_regression": loaded_model_Log, "tree": loaded_model_Tree}
    model_selected = models[model]
    diagnoz_df = pd.read_excel('model_KNN/dataset.xlsx')
    diagnoz_df.drop_duplicates(inplace=True)

    label_encoder = LabelEncoder()
    diagnoz_df["Диагноз"] = label_encoder.fit_transform(diagnoz_df["Диагноз"])
    x = diagnoz_df.drop(["Диагноз"], axis=1)
    y = diagnoz_df["Диагноз"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

    model_selected.fit(x_train, y_train)
    y_pred = model_selected.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {"accuracy": round(accuracy, 3), "precision": round(precision, 3), "recall": round(recall, 3)}
@app.route("/")
def index():
    return render_template('index.html', title="Добро пожаловать на наш сайт!", menu=menu)
@app.route("/model_KNN", methods=['POST', 'GET'])
def func_KNN():
    if request.method == 'GET':
        return render_template('KNN.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    elif request.method == 'POST':
        metrics = classification_model_metrics("knn")
        X_test = np.array([[float(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3'])]])
        pred = loaded_model_knn.predict(X_test)
        diagnoz = diagnoz_df[pred[0]]
        return render_template('KNN.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + diagnoz,
                               accuracy=metrics['accuracy'],
                               precision=metrics['precision'], recall=metrics['recall'])


@app.route("/model_Logistic", methods=['POST', 'GET'])
def func_log():
    if request.method == 'GET':
        return render_template('Logistic.html', title="Логистическая регрессия", menu=menu)
    elif request.method == 'POST':
        metrics = classification_model_metrics("logistic_regression")

        X_test = np.array([[float(request.form['list1']),
                            int(request.form['list2']),
                            int(request.form['list3'])]])

        pred = loaded_model_Log.predict(X_test)
        diagnoz = diagnoz_df[pred[0]]

        return render_template('Logistic.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + diagnoz,
                               accuracy=metrics['accuracy'],
                               precision=metrics['precision'],
                               recall=metrics['recall'])

@app.route("/model_Wood", methods=['GET', 'POST'])
def func_wood():
    if request.method == 'GET':
        return render_template('Wood.html', title="Дерево решений", menu=menu)
    elif request.method == 'POST':
        metrics = classification_model_metrics("tree")
        X_test = np.array([[float(request.form['list1']),
                          int(request.form['list2']),
                          int(request.form['list3'])]])
        pred = loaded_model_Tree.predict(X_test)
        diagnoz = diagnoz_df[pred[0]]
        return render_template('Wood.html', title="Дерево решений", menu=menu,
                               class_model="Это: " + diagnoz,
                               accuracy = metrics['accuracy'],
                               precision = metrics['precision'], recall = metrics['recall'])
@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('temp')),
                       float(request.args.get('pulse')),
                       float(request.args.get('pain_level'))]])
    pred = loaded_model_knn.predict(X_new)
    diagnoz = diagnoz_df[pred[0]]
    return jsonify(sort=diagnoz)

if __name__ == "__main__":
    app.run(debug=True)
