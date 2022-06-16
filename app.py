from crypt import methods
import flask 
from flask import render_template
import pickle
import sklearn
import numpy as np
from sklearn import svm
from tensorflow.keras import layers

app = flask.Flask(__name__, template_folder= 'templates')

@app.route('/', methods=["POST", "GET"])

@app.route('/index', methods=["POST", "GET"])
def main():
    if flask.request.method == "GET":
        return render_template('main.html')

    if flask.request.method == "POST":
        with open('model_svr.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        x1 = float(flask.request.form['plotnost'])
        x2 = float(flask.request.form['modul_upr'])
        x3 = float(flask.request.form['otverditel'])
        x4 = float(flask.request.form['epox_group'])
        x5 = float(flask.request.form['t_vspiski'])
        x6 = float(flask.request.form['pov_plotnost'])
        x7 = float(flask.request.form['potreb_smoli'])
        x8 = float(flask.request.form['_step_nashivki'])
        x9 = float(flask.request.form['plot_nashivki'])
        x10 = float(flask.request.form['ugol_nashivki'])
        x_mu = np.array([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]])
        x_normalizer = layers.Normalization(input_shape=[1,], axis=None)
        x_normalizer.adapt(x_mu)
        x_new = x_normalizer(x_mu).numpy()
        y_pred = loaded_model.predict(x_new)
        y = round(float(y_pred), 3)

        return render_template('main.html', result = y)

if __name__ == '__main__':
    app.run()   