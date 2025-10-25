from flask import Flask, render_template, request
import pandas as pd
from recommender_v2 import (
    convert_price_col_to_float,
    find_top_listings_by_amenities_and_room_type,
    convert_bus_stops_to_latlon,
    compute_bus_proximity_scores,
    compute_poi_liveability_fast,
    calc_crime_rates_by_neigbourhood
)
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
def index():
    
    
    df = pd.read_csv("listings.csv.gz")

    
    room_types = sorted(df['room_type'].dropna().unique())
    importance = [1,2,3,4,5]
    return render_template('index.html', result_amounts=[5,10,15,20], room_types=room_types, crime_importances=importance, transportation_importances=importance, price_importances=importance, liveability_importances=importance)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        
        amenities = request.form.get('amenities')
        amenities = amenities.strip().split(',')
        
        room_type = request.form.get('room_type')
        result_amount = int(request.form.get('result_amount'))
        max_price = int(request.form.get('max_price'))
        
        user_prefs = { #client should be able to select these
        'transportation': int(request.form.get('transportation')),  # importance scale: 1 (low) - 5 (high)
        'crime': int(request.form.get('crime')), 
        'price': int(request.form.get('price')),
        'liveability': int(request.form.get('liveability'))
        }
        
        df = pd.read_csv("listings.csv.gz")
        bus_stops_df = pd.read_csv("bus_stops.csv")
        crime_df = pd.read_csv("BOROUGH.csv")
        crime_df = crime_df[crime_df.BoroughName != 'London Heathrow and London City Airports']
        pois_df = pd.read_csv("london_pois.csv")
        
        convert_price_col_to_float(df)
        df_filtered = find_top_listings_by_amenities_and_room_type(
            amenities, room_type, df, top_n=200, min_reviews=5
        )
        df_filtered = df_filtered[df_filtered['price'] <= max_price]
        
        bus_stops_df = convert_bus_stops_to_latlon(bus_stops_df)
        df_with_transport = compute_bus_proximity_scores(df_filtered, bus_stops_df)
        df_final = compute_poi_liveability_fast(df_with_transport, pois_df)

        
        crime_scores = calc_crime_rates_by_neigbourhood(crime_df)
        df_final = df_final.merge(crime_scores, left_on='neighbourhood_cleansed', right_on='BoroughName')

        if df_final.empty:
            return render_template('error.html') 
        
        scaler = MinMaxScaler()
        features_to_scale = df_final[["poi_liveability_score","transport_score","price","crime_score"]]
        scaled_features = scaler.fit_transform(features_to_scale)
        df_final[["poi_liveability_score_scaled","transportation_score_scaled","price_scaled","crime_score_scaled"]] = scaled_features
        
        total = sum(user_prefs.values())
        weights = {k: v / total for k, v in user_prefs.items()}
        df_final['weighted_score'] = (
        weights['transportation'] * df_final['transportation_score_scaled'] +
        weights['crime'] * (1 - df_final['crime_score_scaled']) +
        weights['price'] * (1 - df_final['price_scaled'])+
        weights['liveability'] * df_final['poi_liveability_score_scaled']
        )

        top_results = df_final.sort_values('weighted_score', ascending=False).head(result_amount).to_dict(orient='records') #client should be able to see the amount of results, maybe at max 20
        print(user_prefs)
        print(top_results) #for debugging
        return render_template('results.html', listings=top_results)

    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == "__main__":
    app.run(debug=True)