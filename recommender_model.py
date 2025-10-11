import nltk
import pandas as pd
import numpy as np
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from ast import literal_eval





def calcWeightedRating(row, avgRating, numOfRatings, minThres, defRating): # New rating based on the amount of reviews
    weightedRating = ((row[avgRating] * row[numOfRatings]) + (minThres * defRating))/(row[numOfRatings] + minThres)
    
    return weightedRating




def recommend(location, tags, df):
    tags = ([i.lower() for i in tags]) # Tags we want to find in the listing
    stop_words = stopwords.words('english')
    filtered_df = df[df['neighbourhood_cleansed'] == location] # We actually want to get all locations within a some radius based on longitude and langitude but this is just a quick implementation
    filtered_df = filtered_df.set_index(np.arange(filtered_df.shape[0]))
    cos = []
    lemmatizer = WordNetLemmatizer()
    
    filtered_set = set()
    for i in tags:
        filtered_set.add(lemmatizer.lemmatize(i))
    for i in range(filtered_df.shape[0]):
        temp_token = word_tokenize(filtered_df["amenities"][i])
        temp_set = [word for word in temp_token if word not in stop_words]
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemmatizer.lemmatize(s))
        vector = temp2_set.intersection(filtered_set)
        cos.append(len(vector))
    filtered_df['similarity'] = cos
    filtered_df = filtered_df.sort_values(by=['similarity','Weighted-Rating'], ascending=False)
    
    return filtered_df[['neighbourhood_cleansed', 'Weighted-Rating', 'listing_url', 'id', 'similarity']].head(5)

def main():
    df = pd.read_csv('listings.csv')

    df_above_10 = df[df['number_of_reviews'] >= 10] # Only look at listings with 10 or more reviews
    df_above_10 = df_above_10.copy()
    
    df_above_10['Weighted-Rating'] = df_above_10.apply(lambda x: calcWeightedRating(x, 'review_scores_value', 'number_of_reviews', 10, 2.5),axis=1)
    df_above_10['amenities'] = df_above_10['amenities'].str.lower().str.strip('[]').str.replace(',', '').str.replace('"', '') # amenities doesn't really want to work unless its provided as a long string like this
    
    print(recommend('Lewisham', ['glasses', 'breakfast','Cooking basics']))
    
if __name__=="__main__":
    main()