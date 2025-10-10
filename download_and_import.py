import sqlite3
import pandas as pd
import gzip
import os

DB_PATH = "airbnb_london.db"
CSV_PATH = "listings.csv"

df = pd.read_csv(CSV_PATH)

def clean_and_prepare(df: pd.DataFrame):
    pref = [
        'id','name','neighbourhood_cleansed','latitude','longitude',
        'room_type','price','minimum_nights','number_of_reviews',
        'review_scores_rating','amenities','instant_bookable'
    ]
    available = [c for c in pref if c in df.columns]
    df = df[available].copy()
    if 'price' in df.columns:
        df['price'] = df['price'].astype(str).str.replace('[^0-9.]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
    df['review_scores_rating'] = pd.to_numeric(df.get('review_scores_rating', None), errors='coerce')
    df['number_of_reviews'] = pd.to_numeric(df.get('number_of_reviews', 0), errors='coerce').fillna(0).astype(int)
    return df

df = clean_and_prepare(df)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
conn = sqlite3.connect(DB_PATH)
df.to_sql("listings", conn, if_exists="replace", index=False)
conn.close()
