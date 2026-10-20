import pickle
import re

import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__, template_folder="./templates", static_folder="./static")

with open("model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("vector.pkl", "rb") as vector_file:
    vector = pickle.load(vector_file)

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words("english"))
corpus = []


def fake_news_det(news):
    review = news
    review = re.sub(r"[^a-zA-Z\s]", "", review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review:
        if y not in stpwrds:
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [" ".join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)

    return prediction


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        message = request.form["news"]
        pred = fake_news_det(message)
        print(pred)

        def predi(pred):
            if pred[0] == 1:
                res = "Prediction of the News :  Looking Fake News📰"
            else:
                res = "Prediction of the News : Looking Real News📰 "
            return res

        result = predi(pred)
        return render_template("prediction.html", prediction_text=f"{result}")
    else:
        return render_template("prediction.html", prediction="Something went wrong")


if __name__ == "__main__":
    app.run(debug=True)
