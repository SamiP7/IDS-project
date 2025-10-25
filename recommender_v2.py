from pyproj import Transformer
import numpy as np
import pandas as pd
from typing import Optional
import ast
from sklearn.neighbors import KDTree


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
    crime_rates_df['crime_score'] = crime_rates_df.sum(axis=1, numeric_only=True)
    crime_rates_df[['MajorText','BoroughName','crime_score']]

    for i in weights.keys():
        crime_rates_df['crime_score'] = np.where(crime_rates_df['MajorText'] == i, crime_rates_df['crime_score']*weights[i], crime_rates_df['crime_score'])
    for i in pop_by_borough.keys():
        crime_rates_df['crime_score'] = np.where(crime_rates_df['BoroughName'] == i, crime_rates_df['crime_score']/pop_by_borough[i], crime_rates_df['crime_score'])
    crime_combined = crime_rates_df.groupby(['BoroughName']).sum()
    crime_combined.sum(axis=1, numeric_only=True)
    normalized = (crime_combined['crime_score']-crime_combined['crime_score'].min())/(crime_combined['crime_score'].max()-crime_combined['crime_score'].min())
    
    return pd.DataFrame(normalized)
    
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

def compute_bus_proximity_scores_fast(listings, bus_stops, radius_m=1000, near_threshold_m=200):
    bus_coords = np.radians(bus_stops[["latitude", "longitude"]].to_numpy())
    listing_coords = np.radians(listings[["latitude", "longitude"]].to_numpy())

    tree = KDTree(bus_coords, leaf_size=16)
    
    dist, idx = tree.query(listing_coords, k=1)
    nearest_bus_stop_m = dist * 6371000  # convert rad to meters

        # count within radius
    counts_within = np.array([len(tree.query_ball_point(x, radius_m / 6371000)) for x in listing_coords])
    
    listings["nearest_bus_stop_m"] = nearest_bus_stop_m
    listings[f"n_bus_stops_within_{radius_m}m"] = counts_within
    listings["transport_score"] = np.maximum(0, (radius_m - nearest_bus_stop_m) / radius_m + (counts_within / counts_within.max()))
    bonus = (nearest_bus_stop_m < near_threshold_m).astype(int) * 0.5
    listings["transport_score"] += bonus
    return listings


def compute_poi_liveability_fast(listings: pd.DataFrame, pois: pd.DataFrame,
                                 radius_m: int = 500,
                                 poi_weights: dict = None) -> pd.DataFrame:
    """
    Compute a POI liveability score for each listing based on nearby POIs.
    
    Parameters:
    - listings: DataFrame with 'latitude' and 'longitude'
    - pois: DataFrame with 'lat', 'lon', 'category'
    - radius_m: search radius in meters
    - poi_weights: dict of {category: weight}, default provided if None
    
    Returns:
    - listings DataFrame with added 'poi_liveability_score' column
    """
    if poi_weights is None:
        poi_weights = {
            "restaurant": 1.0,
            "cafe": 0.8,
            "bar": 0.6,
            "pub": 0.6,
            "fast_food": 0.4,
            "supermarket": 1.2,
            "convenience": 0.6,
            "bakery": 0.7,
        }
    
    # Convert coordinates to radians
    listing_coords = np.radians(listings[["latitude", "longitude"]].to_numpy())
    poi_coords = np.radians(pois[["lat", "lon"]].to_numpy())
    poi_categories = pois["category"].to_numpy()
    
    earth_radius = 6371000  # meters
    n_listings = len(listings)
    poi_score_raw = np.zeros(n_listings)
    
    # Compute contribution per category
    for cat, weight in poi_weights.items():
        idx_cat = np.where(poi_categories == cat)[0]
        if len(idx_cat) == 0:
            continue
        tree_cat = cKDTree(poi_coords[idx_cat])
        counts = tree_cat.query_ball_point(listing_coords, r=radius_m / earth_radius)
        for i, pts in enumerate(counts):
            if len(pts) > 0:
                dists = np.linalg.norm(listing_coords[i] - poi_coords[idx_cat][pts], axis=1)
                contrib = np.sum(weight * (1 - dists / (radius_m / earth_radius)))
                poi_score_raw[i] += contrib
    
    # Normalize 0..1
    poi_score_norm = poi_score_raw / poi_score_raw.max() if poi_score_raw.max() > 0 else poi_score_raw
    listings = listings.copy()
    listings["poi_liveability_score"] = poi_score_norm
    return listings   


                                                                                                                                            
