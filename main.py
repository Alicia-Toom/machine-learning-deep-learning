from flask import Flask, render_template
from flask import request
from services.prediction_service import PredictionService

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form.get("ticker")

    prediction_service = PredictionService(ticker)
    prediction_service.train()
    prediction = prediction_service.predict()
    return render_template("prediction.html", prediction=prediction)


if __name__ == '__main__':
    app.run()
