import pickle
import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

diagnoz_df = {
    0: "Не болен",
    1: "Болен"
}

menu = [{"name": "KNN", "url": "model_KNN"},
        {"name": "Logistic", "url": "model_Logistic"},
        {"name": "Wood", "url": "model_Wood"}]

loaded_model_knn = pickle.load(open('model_KNN/model_KNN', 'rb'))
loaded_model_Log = pickle.load(open('model_logistic/model_logistic', 'rb'))
loaded_model_Tree = pickle.load(open('model_wood/model_wood', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', title="Добро пожаловать на наш сайт!", menu=menu)


@app.route("/model_KNN", methods=['POST', 'GET'])
def func_KNN():
    if request.method == 'GET':
        return render_template('KNN.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    elif request.method == 'POST':
        X_test = np.array([[float(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3'])]])
        pred = loaded_model_knn.predict(X_test)
        diagnoz = diagnoz_df[pred[0]]
        return render_template('KNN.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + diagnoz)

@app.route("/model_Logistic", methods=['POST', 'GET'])
def func_log():
    if request.method == 'GET':
        return render_template('Logistic.html', title="Логистическая регрессия", menu=menu)
    elif request.method == 'POST':
        X_test = np.array([[float(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3'])]])
        pred = loaded_model_Log.predict(X_test)
        diagnoz = diagnoz_df[pred[0]]
        return render_template('Logistic.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + diagnoz)

@app.route("/model_Wood", methods=['GET', 'POST'])
def func_wood():
    if request.method == 'GET':
        return render_template('Wood.html', title="Дерево решений", menu=menu)
    elif request.method == 'POST':
        X_test = np.array([[float(request.form['list1']),
                          int(request.form['list2']),
                          int(request.form['list3'])]])
        pred = loaded_model_Tree.predict(X_test)
        diagnoz = diagnoz_df[pred[0]]
        return render_template('Wood.html', title="Дерево решений", menu=menu,
                               class_model="Это: " + diagnoz)

if __name__ == "__main__":
    app.run(debug=True)
