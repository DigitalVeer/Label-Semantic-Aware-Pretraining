import pandas as pd
import os

CURR_PATH = os.path.dirname( os.path.abspath( __file__ ) )
DATA_PATH = os.path.join( CURR_PATH, 'data' )

if not os.path.exists( DATA_PATH ):
    os.mkdir( DATA_PATH )

data_paths = [ 'train.csv', 'validation.csv', 'test.csv']
data_paths = { k: os.path.join( CURR_PATH, k ) for k in data_paths}

label_mapping = {
    "AddToPlaylist": "Add To Playlist",
    "BookRestaurant": "Book Restaurant",
    "GetWeather": "Get Weather",
    "PlayMusic": "Play Music",
    "RateBook": "Rate Book",
    "SearchCreativeWork": "Search Creative Work",
    "SearchScreeningEvent": "Search Screening Event"
}

def clean_label( label ):
    return f"{ label_mapping[ label ].strip() }."

train = pd.read_csv( data_paths[ 'train.csv' ] )
val   = pd.read_csv( data_paths[ 'validation.csv' ] )
test  = pd.read_csv( data_paths[ 'test.csv' ] )

def process(df):
    df = df.rename( columns = { "utterance": "text", "label":"intent" } )
    df = df[ [ "text", "intent" ] ]
    df[ "intent" ] = df[ "intent" ].apply( clean_label )
    return df

#concatenate all data
combined = pd.concat( [ train, val, test ] )

#Fix Intent Labels
combined = process( combined )
combined.to_csv(f"{DATA_PATH}/combined.csv")

train = process( train )
val   = process( val )
test  = process( test )

train.to_csv(f"{DATA_PATH}/train.csv", index=False)
val.to_csv(f"{DATA_PATH}/val.csv", index=False)
test.to_csv(f"{DATA_PATH}/test.csv", index=False)