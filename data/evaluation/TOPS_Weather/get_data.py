import pandas as pd
import os

CURR_PATH = os.path.dirname( os.path.abspath( __file__ ) )
DATA_PATH = os.path.join( CURR_PATH, 'data' )

if not os.path.exists( DATA_PATH ):
    os.mkdir( DATA_PATH )

data_paths = [ 'weather_train.tsv', 'weather_eval.tsv', 'weather_test.tsv']
data_paths = { k: os.path.join( CURR_PATH, k ) for k in data_paths}

def read_tsv( path ):
    """
    Read a tab-separated file with a header row.
    """
    return pd.read_csv( path, sep = "\t" )

def process( df ):
    #extract intent
    def extract_intent( row ):
        """Extracts intent from input format.
        i.e. "IN:GET_WEATHER Can you find me reminders of the event" -> "GET_WEATHER"
        """
        return row['semantic_parse'][row['semantic_parse'].find('IN:')+3:row['semantic_parse'].find(' ')]

    def correct_intent( row ):
        #Turn 'GET_WEATHER' into 'Get Weather.'
        return row['intent'].replace( '_', ' ' ).title() + '.'
    
    df = df.rename( columns = { 'utterance': 'text' } )
    df['intent'] = df.apply( extract_intent, axis = 1 )
    df['intent'] = df.apply( correct_intent, axis = 1 )
    df = df.drop( columns = ['semantic_parse', 'domain'] )
    return df

train = read_tsv( data_paths['weather_train.tsv'] )
val   = read_tsv( data_paths['weather_eval.tsv'] )
test  = read_tsv( data_paths['weather_test.tsv'] )

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