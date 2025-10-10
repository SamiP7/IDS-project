from flask import Flask, render_template, request, g
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "airbnb_london.db"
app = Flask(__name__)

CSV_PATH = "listings.csv"

df = pd.read_csv(CSV_PATH)


@app.route("/")
def index():
    neighbourhoods = sorted(df["neighbourhood_cleansed"].dropna().unique())
    room_types = sorted(df["room_type"].dropna().unique())
    return render_template("index.html", neighbourhoods=neighbourhoods, room_types=room_types)

@app.route("/results", methods=["POST"])
def results():
    user_input = {        "latitude": float(request.form.get("latitude") or 51.5072),
        "longitude": float(request.form.get("longitude") or -0.1276),
        "room_type": request.form.get("room_type") or "Entire home/apt",
        "neighbourhood_cleansed": request.form.get("neighbourhood") or "Westminster",
        "review_scores_rating": float(request.form.get("min_rating") or 80),
        "number_of_reviews": float(request.form.get("min_reviews") or 10),
        "price": float(request.form.get("max_price") or 150)
    }

    user_num = np.array([[user_input["latitude"], user_input["longitude"],
                          user_input["review_scores_rating"], user_input["number_of_reviews"], user_input["price"]]])
    user_num_scaled = scaler.transform(user_num)

    user_cat = pd.DataFrame([[user_input["room_type"], user_input["neighbourhood_cleansed"]]],
                            columns=["room_type", "neighbourhood_cleansed"])
    user_cat_enc = encoder.transform(user_cat)

    user_vector = np.hstack([user_num_scaled, user_cat_enc])
    sims = cosine_similarity(user_vector, X)[0]
    df["similarity"] = sims

    results = df.sort_values("similarity", ascending=False).head(20)

    return render_template("results.html", results=results.to_dict(orient="records"), count=len(results))
