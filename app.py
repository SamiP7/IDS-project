from flask import Flask, render_template, request
import pandas as pd
from recommender_v2 import (
    convert_price_col_to_float,
    find_top_listings_by_amenities_and_room_type,
    convert_bus_stops_to_latlon,
    compute_bus_proximity_scores,
    compute_poi_liveability,
    calc_crime_rates_by_neigbourhood
)

app = Flask(__name__)

@app.route('/')
def index():
    
    df = pd.read_csv("listings.csv")

    
    room_types = sorted(df['room_type'].dropna().unique())

    return render_template('index.html', room_types=room_types)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        
        amenities = request.form.getlist('amenities')
        room_type = request.form.get('room_type')

        
        df = pd.read_csv("listings.csv")
        bus_stops_df = pd.read_csv("bus_stops.csv")
        crime_df = pd.read_csv("BOROUGH.csv")

        
        convert_price_col_to_float(df)
        df_filtered = find_top_listings_by_amenities_and_room_type(
            amenities, room_type, df, top_n=200, min_reviews=5
        )

        
        bus_stops_df = convert_bus_stops_to_latlon(bus_stops_df)
        df_with_transport = compute_bus_proximity_scores(df_filtered, bus_stops_df)
        df_final = compute_poi_liveability(df_with_transport)

        
        crime_scores = calc_crime_rates_by_neigbourhood(crime_df)
        df_final = df_final.merge(crime_scores, left_on='neighbourhood_cleansed', right_on='BoroughName')

        if df_final.empty:
            return render_template('error.html') 

        top_5 = df_final.sort_values('review_scores_rating', ascending=False).head(5)
        return render_template('results.html', results=top_5.to_dict(orient='records'))

    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == "__main__":
    app.run(debug=True)

