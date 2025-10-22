import pyproj
from pyproj import Transformer
import numpy as np
import pandas as pd
from typing import Optional
import requests
import time
from collections import Counter
import ast


# parameters
EARTH_R = 6371000.0  # meters
BATCH_SIZE = 5000
NEAR_THRESHOLD_M = 200       # "nearest" considered very close
MORE_OPTIONS_RADIUS_M = 1000 # radius to count additional nearby stops
MIN_NEARBY_STOPS = 3        # require at least this many stops inside MORE_OPTIONS_RADIUS_M



# convert price column to float
def convert_price_col_to_float(df, price_col="price", inplace=True):
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)


def _parse_amenities(val):
    """Return a set of cleaned, lower-case amenity strings from the dataframe cell."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    if isinstance(val, (list, set, tuple)):
        items = val
    else:
        try:
            items = ast.literal_eval(val)
        except Exception:
            # fallback: split on comma and strip punctuation
            items = [s.strip().strip('"\']') for s in str(val).split(',')]
    cleaned = {str(s).strip().lower() for s in items if s is not None and str(s).strip() != ""}
    return cleaned

def calc_crime_rates_by_neigbourhood(crime_rates_df):
    weights = {
        'ARSON AND CRIMINAL DAMAGE': 0.8,
        'BURGLARY': 0.8,
        'DRUG OFFENCES': 0.8,
        'FRAUD AND FORGERY': 0.6,
        'MISCELLANEOUS CRIMES AGAINST SOCIETY': 0.4,
        'POSSESSION OF WEAPONS': 1.0,
        'PUBLIC ORDER OFFENCES': 0.4,
        'ROBBERY': 0.8,
        'SEXUAL OFFENCES': 1.0,
        'THEFT': 0.8,
        'VEHICLE OFFENCES': 0.4,
        'VIOLENCE AGAINST THE PERSON': 1.2,
        'NFIB FRAUD': 0.4
    }
    pop_by_borough = {'Croydon':410000,
                  'Barnet':405000,
                  'Ealing':386000,
                  'Newham':375000,
                  'Brent':353000,
                  'Wandsworth':338000,
                  'Bromley':335000,
                  'Tower Hamlets':332000,
                  'Hillingdon':330000,
                  'Enfield':327000,
                  'Redbridge':321000,
                  'Lambeth':317000,
                  'Southwark':315000,
                  'Lewisham':301000,
                  'Greenwich':300000,
                  'Hounslow':300000,
                  'Waltham Forest':280000,
                  'Havering':276000,
                  'Harrow':271000,
                  'Hackney':267000,
                  'Haringey':264000,
                  'Bexley':256000,
                  'Barking and Dagenham':233000,
                  'Islington':223000,
                  'Merton':219000,
                  'Camden':217000,
                  'Sutton':215000,
                  'Westminster':210000,
                  'Richmond upon Thames':197000,
                  'Hammersmith and Fulham':189000,
                  'Kingston upon Thames':173000,
                  'Kensington and Chelsea':145000,
                  'City of London':15000
    } #SOURCE FOR NUMBERS: https://www.citypopulation.de/en/uk/greaterlondon/
    crime_rates_df = crime_rates_df.replace(to_replace='Unknown', value='City of London')
    crime_rates_df['total'] = crime_rates_df.sum(axis=1, numeric_only=True)
    crime_rates_df[['MajorText','BoroughName','total']]

    for i in weights.keys():
        crime_rates_df['total'] = np.where(crime_rates_df['MajorText'] == i, crime_rates_df['total']*weights[i], crime_rates_df['total'])
    for i in pop_by_borough.keys():
        crime_rates_df['total'] = np.where(crime_rates_df['BoroughName'] == i, crime_rates_df['total']/pop_by_borough[i], crime_rates_df['total'])
    crime_combined = crime_rates_df.groupby(['BoroughName']).sum()
    crime_combined.sum(axis=1, numeric_only=True)
    normalized = (crime_combined['total']-crime_combined['total'].min())/(crime_combined['total'].max()-crime_combined['total'].min())
    
    return normalized
    
def find_top_listings_by_amenities_and_room_type(required_amenities, room_type, df_source, top_n=100, min_reviews=5):
    """
    Filter listings that contain all required_amenities (case-insensitive, substring match),
    then rank by a composite score of review rating * log1p(number_of_reviews).
    Returns top_n rows from df_source.
    - required_amenities: list or str (single amenity). Example: ['wifi', 'washer'] or 'Wifi'
    - min_reviews: minimum number_of_reviews to consider (set 0 to disable)
    """
    # filter by room type
    df_source = df_source[df_source["room_type"].str.strip().str.lower() == room_type.strip().lower()]

    # normalize input
    
    if isinstance(required_amenities, str):
        required = [required_amenities]
    else:
        required = list(required_amenities)
    required = [r.strip().lower() for r in required if r and str(r).strip() != ""]

    if len(required) == 0:
        raise ValueError("Please provide at least one amenity to filter by.")

    def has_all(amen_cell):
        aset = _parse_amenities(amen_cell)
        # require each requested amenity to appear as a substring in at least one amenity entry
        for req in required:
            if not any(req in a for a in aset):
                return False
        return True

    # apply filter
    mask = df_source["amenities"].apply(has_all)
    df_filtered = df_source[mask].copy()

    # enforce minimum reviews
    if min_reviews and min_reviews > 0:
        df_filtered = df_filtered[df_filtered["number_of_reviews"].fillna(0) >= min_reviews]

    if df_filtered.empty:
        return df_filtered  # nothing found

    # composite score: rating * log1p(num_reviews)
    rating = df_filtered["review_scores_rating"].fillna(0).astype(float)
    nrev = df_filtered["number_of_reviews"].fillna(0).astype(float)
    df_filtered["_composite_score"] = rating * np.log1p(nrev)

    # sort by composite, then rating, then number_of_reviews
    df_sorted = df_filtered.sort_values(
        by=["_composite_score", "review_scores_rating", "number_of_reviews"],
        ascending=[False, False, False]
    )

    # drop helper column before returning (but keep if you want to inspect)
    return df_sorted.head(top_n).drop(columns=["_composite_score"], errors="ignore")

def filter_listings_by_neighborhood(df_listings, neighborhood_name, neighborhood_col="neighbourhood_cleansed"):
    """
    Filter listings DataFrame to only include rows where the neighborhood column matches the given neighborhood name.
    
    Parameters:
    df_listings (pd.DataFrame): DataFrame containing listing data with a neighborhood column.
    neighborhood_name (str): The name of the neighborhood to filter by.
    neighborhood_col (str): The name of the column in df_listings that contains neighborhood names.
    
    Returns:
    pd.DataFrame: Filtered DataFrame containing only listings in the specified neighborhood.
    """
    mask = df_listings[neighborhood_col].str.strip().str.lower() == neighborhood_name.strip().lower()
    filtered_df = df_listings[mask].copy()
    return filtered_df


def convert_bus_stops_to_latlon(bus_stops):
    """
    Convert bus stop coordinates from Easting/Northing (OSGB36) to latitude/longitude (WGS84).
    
    Parameters:
    bus_stops (pd.DataFrame): DataFrame containing bus stop data with 'Location_Easting' and 'Location_Northing' columns.
    
    Returns:
    pd.DataFrame: DataFrame with added 'latitude' and 'longitude' columns.
    """
    # EPSG:27700 = OSGB36 / British National Grid, EPSG:4326 = WGS84 lat/lon
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    # transform vectors of Easting (x) and Northing (y) -> (lon, lat)
    lon, lat = transformer.transform(bus_stops["Location_Easting"].values, bus_stops["Location_Northing"].values)

    bus_stops["latitude"] = lat
    bus_stops["longitude"] = lon

    return bus_stops



def compute_bus_proximity_scores(
    listings: pd.DataFrame,
    bus_stops: pd.DataFrame,
    bus_lat: Optional[np.ndarray] = None,
    bus_lon: Optional[np.ndarray] = None,
    more_options_radius_m: int = None,
    near_threshold_m: int = None,
    batch_size: int = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    For each listing in `listings` compute:
      - nearest_bus_stop_m: distance (m) to nearest bus stop
      - n_bus_stops_within_radius: number of bus stops within more_options_radius_m
      - bus_proximity_score_raw: sum of linear proximity weights from stops within radius
      - bus_proximity_score: normalized score 0..1 (if normalize=True)
      - nearest_bus_stop_idx: index into bus_stops of nearest stop

    The proximity weight used is w = max(0, (R - d) / R) for stops within R meters.
    Closer stops contribute more; more stops increase the score.

    This function is vectorized and processes listings in batches to limit memory use.
    """
    # fall back to notebook constants if provided
    try:
        if more_options_radius_m is None:
            more_options_radius_m = MORE_OPTIONS_RADIUS_M
        if near_threshold_m is None:
            near_threshold_m = NEAR_THRESHOLD_M
        if batch_size is None:
            batch_size = BATCH_SIZE
    except NameError:
        # if notebook constants not available, use defaults
        if more_options_radius_m is None:
            more_options_radius_m = 1000
        if near_threshold_m is None:
            near_threshold_m = 200
        if batch_size is None:
            batch_size = 5000

    # Obtain bus stop coordinate arrays
    if bus_lat is None or bus_lon is None:
        if "latitude" in bus_stops.columns and "longitude" in bus_stops.columns:
            bus_lat_arr = bus_stops["latitude"].values.astype(float)
            bus_lon_arr = bus_stops["longitude"].values.astype(float)
        else:
            raise ValueError("bus_lat/bus_lon not provided and bus_stops has no latitude/longitude")
    else:
        bus_lat_arr = np.asarray(bus_lat, dtype=float)
        bus_lon_arr = np.asarray(bus_lon, dtype=float)

    if "latitude" not in listings.columns or "longitude" not in listings.columns:
        raise ValueError("listings must have 'latitude' and 'longitude' columns")

    n_listings = len(listings)
    n_stops = len(bus_lat_arr)

    # prepare result arrays
    nearest_dists = np.full(n_listings, np.nan, dtype=float)
    nearest_idx = np.full(n_listings, -1, dtype=int)
    counts_within = np.zeros(n_listings, dtype=int)
    raw_scores = np.zeros(n_listings, dtype=float)

    # vectorized haversine that supports broadcasting (lat1 can be shape (m,1), lat2 shape (1,k))
    def haversine_matrix(lat1, lon1, lat2, lon2):
        # all inputs in degrees
        lat1r = np.radians(lat1)
        lon1r = np.radians(lon1)
        lat2r = np.radians(lat2)
        lon2r = np.radians(lon2)
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return EARTH_R * c

    # process in batches
    lat_vals = listings["latitude"].values.astype(float)
    lon_vals = listings["longitude"].values.astype(float)

    for start in range(0, n_listings, batch_size):
        end = min(start + batch_size, n_listings)
        lat_batch = lat_vals[start:end]  # shape (m,)
        lon_batch = lon_vals[start:end]

        # broadcast to (m, k) to get matrix of distances: each row -> distances to all bus stops
        dmat = haversine_matrix(lat_batch[:, None], lon_batch[:, None], bus_lat_arr[None, :], bus_lon_arr[None, :])
        # nearest distances and indices for this batch
        local_min_idx = np.argmin(dmat, axis=1)
        local_min_d = dmat[np.arange(dmat.shape[0]), local_min_idx]

        # counts within radius
        within_mask = dmat <= more_options_radius_m
        local_counts = within_mask.sum(axis=1)

        # raw score: sum of linear weights for stops within radius: (R - d)/R
        weights = np.clip((more_options_radius_m - dmat) / more_options_radius_m, 0.0, 1.0)
        local_raw_scores = weights.sum(axis=1)

        # write back
        nearest_dists[start:end] = local_min_d
        nearest_idx[start:end] = local_min_idx
        counts_within[start:end] = local_counts
        raw_scores[start:end] = local_raw_scores

    # optional near stop bonus: listings with any stop closer than near_threshold get +1 to raw score
    near_bonus = (nearest_dists <= near_threshold_m).astype(float)
    raw_scores += near_bonus  # small boost for having a very nearby stop



    # assemble results into a DataFrame (or attach to listings copy)
    res = listings.copy()
    res["nearest_bus_stop_m"] = nearest_dists
    res["nearest_bus_stop_idx"] = nearest_idx
    res[f"n_bus_stops_within_{more_options_radius_m}m"] = counts_within
    res["transport_score"] = raw_scores
    
    return res

# Uses the free Overpass (OpenStreetMap) API to count nearby shops/restaurants and compute a simple liveability score.
# Adds columns to df_with_transport_scores. Be careful with large numbers of listings (Overpass rate limits) —
# use max_listings to limit queries or rely on caching via rounding.


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# POI categories we will count and their Overpass tag filters
POI_CATEGORY_QUERIES = {
    "restaurant": '["amenity"="restaurant"]',
    "cafe": '["amenity"="cafe"]',
    "bar": '["amenity"="bar"]',
    "pub": '["amenity"="pub"]',
    "fast_food": '["amenity"="fast_food"]',
    "supermarket": '["shop"="supermarket"]',
    "convenience": '["shop"="convenience"]',
    "bakery": '["shop"="bakery"]',
    # generic shops (counts all shop=* excluding the specific ones above)
    "shop": '["shop"]',
}

def _build_overpass_query(lat, lon, radius_m=500):
    """
    Build a query that fetches nodes/ways/relations for our POI filters around a point.
    """
    around = f'(around:{int(radius_m)},{lat},{lon})'
    parts = []
    for tag in POI_CATEGORY_QUERIES.values():
        # nodes, ways and relations
        parts.append(f'node{tag}{around};')
        parts.append(f'way{tag}{around};')
        parts.append(f'relation{tag}{around};')
    # combine and request tags
    q = "[out:json][timeout:25];(" + "".join(parts) + ");out center tags;"
    return q


def query_pois(lat, lon, radius_m=500, retry=3, pause=1.0):
    """
    Query Overpass and return a Counter of category -> count for the POI_CATEGORY_QUERIES keys.
    Returns an empty Counter on persistent failure.
    """
    q = _build_overpass_query(lat, lon, radius_m)
    for attempt in range(retry):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": q}, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                counts = Counter()
                for el in data.get("elements", []):
                    tags = el.get("tags") or {}
                    # determine which categories this element contributes to
                    for cat, tag_filter in POI_CATEGORY_QUERIES.items():
                        # simple check: see if element matches tag for this category
                        if cat in ("shop",):  # "shop" generic: any shop tag counts, but exclude bakery/convenience/supermarket duplicates later
                            if "shop" in tags:
                                counts["shop"] += 1
                        else:
                            # for amenities we check exact tag presence in tags
                            if cat in ("restaurant", "cafe", "bar", "pub", "fast_food"):
                                if tags.get("amenity") == cat:
                                    counts[cat] += 1
                            else:
                                # shop subtypes
                                if tags.get("shop") == cat:
                                    counts[cat] += 1
                # adjust shop generic to not double-count specific shop types:
                # subtract bakery, convenience, supermarket from generic shop count if present
                specific_shops = counts.get("bakery", 0) + counts.get("convenience", 0) + counts.get("supermarket", 0)
                if counts.get("shop", 0) > specific_shops:
                    counts["shop_generic"] = counts["shop"] - specific_shops
                else:
                    counts["shop_generic"] = 0
                # ensure all keys exist
                result = {k: int(counts.get(k, 0)) for k in ["restaurant","cafe","bar","pub","fast_food","supermarket","convenience","bakery","shop_generic"]}
                return result
            else:
                time.sleep(pause)
        except Exception:
            time.sleep(pause)
    # failure -> return zeros
    return {k: 0 for k in ["restaurant","cafe","bar","pub","fast_food","supermarket","convenience","bakery","shop_generic"]}



def compute_poi_liveability(
    listings_df,
    lat_col="latitude",
    lon_col="longitude",
    radius_m=500,
    rounding_deg=0.001,
    max_listings=None,
    sleep_between_requests=1.0,
):
    """
    For each listing compute nearby POI counts and a normalized liveability score.
    - rounding_deg: round coordinates to this degree precision for caching (0.001 ~ 100-110m).
    - max_listings: optional limit to number of listings to query (useful to avoid hitting API limits).
    Returns a new DataFrame with added columns:
      n_restaurant, n_cafe, n_bar, n_pub, n_fast_food, n_supermarket, n_convenience, n_bakery, n_shop_generic,
      poi_liveability_raw, poi_liveability_score
    """
    coords = list(zip(listings_df[lat_col].values, listings_df[lon_col].values))
    n = len(coords) if max_listings is None else min(len(coords), int(max_listings))

    cache = {}
    results = []
    for idx in range(n):
        lat, lon = coords[idx]
        key = (round(float(lat)/rounding_deg)*rounding_deg, round(float(lon)/rounding_deg)*rounding_deg)
        if key in cache:
            counts = cache[key]
        else:
            counts = query_pois(lat, lon, radius_m=radius_m)
            cache[key] = counts
            time.sleep(sleep_between_requests)
        results.append(counts)

    # create DataFrame of counts (for the processed rows)
    counts_df = pd.DataFrame(results)
    # compute a raw score as weighted sum (you can tune weights)
    weights = {
        "restaurant": 1.0,
        "cafe": 0.8,
        "bar": 0.6,
        "pub": 0.6,
        "fast_food": 0.4,
        "supermarket": 1.2,
        "convenience": 0.6,
        "bakery": 0.7,
        "shop_generic": 0.5,
    }
    counts_df["poi_liveability_raw"] = sum(counts_df[col] * w for col, w in weights.items())
    # normalize to 0..1 based on observed max in the processed subset
    max_raw = counts_df["poi_liveability_raw"].max() if len(counts_df) > 0 else 1.0
    if max_raw <= 0:
        counts_df["poi_liveability_score"] = 0.0
    else:
        counts_df["poi_liveability_score"] = counts_df["poi_liveability_raw"] / float(max_raw)

    # merge results back into a copy of the listings dataframe
    out = listings_df.copy()
    # set default zeros for all new columns first
    for col in ["restaurant","cafe","bar","pub","fast_food","supermarket","convenience","bakery","shop_generic","poi_liveability_raw","poi_liveability_score"]:
        out[col if col.startswith("n_") == False and col not in ("poi_liveability_raw","poi_liveability_score") else col] = 0.0

    # assign only for processed rows
    assign_df = counts_df.rename(columns={
        "restaurant":"n_restaurant",
        "cafe":"n_cafe",
        "bar":"n_bar",
        "pub":"n_pub",
        "fast_food":"n_fast_food",
        "supermarket":"n_supermarket",
        "convenience":"n_convenience",
        "bakery":"n_bakery",
        "shop_generic":"n_shop_generic",
    })
    # ensure index alignment
    assign_df.index = out.index[:len(assign_df)]
    out.loc[assign_df.index, assign_df.columns] = assign_df

    return out

def main():

    df = pd.read_csv("listings.csv", usecols=[
        "id","name","price","amenities","neighbourhood_cleansed",
        "room_type","number_of_reviews","review_scores_rating","latitude","longitude"
    ])
    bus_stops_df = pd.read_csv("bus_stops.csv")
    user_prefs = {
    'transportation': 5,  # importance scale: 1 (low) - 5 (high)
    'crime': 3,
    'amenities': 4, # we can delete amenities weights as we are filtering already
    'price': 4
}
    
    crime_df = pd.read_csv('BOROUGH.csv')
    crime_rates_by_neighourhood = calc_crime_rates_by_neigbourhood(crime_df)
    
    wanted_amenities = ['wifi', 'washer'] # example will be replaced by user input
    neighborhood_name = 'greenwich' # example will be replaced by user input
    room_type = 'entire home/apt' # example will be replaced by user input

    convert_price_col_to_float(df)
    df_top_amenities = find_top_listings_by_amenities_and_room_type(wanted_amenities,room_type, df, top_n=200, min_reviews=5)
    df_top_amenities = filter_listings_by_neighborhood(df_top_amenities, neighborhood_name) # we can delete this part if it makes crime score meaningless
  
    bus_stops_df = convert_bus_stops_to_latlon(bus_stops_df)
    df_with_transport_scores = compute_bus_proximity_scores(
        listings=df_top_amenities,
        bus_stops=bus_stops_df,
    )
    df_final = compute_poi_liveability(
        df_with_transport_scores,
        radius_m=500,
        rounding_deg=0.001,
        max_listings=200,
        sleep_between_requests=1.0,
    )


    # Normalize the weights
    total = sum(user_prefs.values())
    weights = {k: v / total for k, v in user_prefs.items()}


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_to_scale = df_final[["poi_liveability_score","transport_score","price"]] # "crime_score" will be added
    scaled_features = scaler.fit_transform(features_to_scale)
    df_final[["amenities_score_scaled","transportation_score_scaled","price_scaled"]] = scaled_features # crime score_scaled will be added

    df_final['weighted_score'] = (
    weights['transportation'] * df_final['transportation_score_scaled'] +
    #weights['crime'] * (1 - df_final['crime_score_scaled']) +  # lower crime = better -- will be added
    weights['amenities'] * df_final['amenities_score_scaled'] + # will be deleted if we dont use weights for amenities
    weights['price'] * (1 - df_final['price_scaled'])
)

    top_5 = df_final.sort_values('weighted_score', ascending=False).head(5)
    print(top_5[["id","name","price","amenities","room_type","neighbourhood_cleansed","weighted_score","poi_liveability_score","transport_score"]]) # crime_score column will be added, scores will not be displayed
                                                                                                                                                    # they are for checking the scores
    

if __name__ == "__main__":
    main()



