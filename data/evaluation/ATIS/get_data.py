import pandas as pd
import os
from sklearn.model_selection import train_test_split

CURR_PATH = os.path.dirname( os.path.abspath( __file__ ) )
DATA_PATH = os.path.join( CURR_PATH, 'data' )

if not os.path.exists( DATA_PATH ):
    os.mkdir( DATA_PATH )

data_paths = [ 'train.json', 'test.json']
data_paths = { k: os.path.join( CURR_PATH, k ) for k in data_paths}

#Read jsons and combine to concatenated dataframe
cdf = pd.DataFrame()
for k in data_paths:
    df = pd.read_json( data_paths[k] )
    cdf = pd.concat( [ cdf, df ] )

def process( df ):
  df.drop('slots', axis=1, inplace=True)

  def correct_label( row ):
    return " ".join( [word.capitalize() for word in row.split("_")] ) + "."

  df['intent'] = df['intent'].apply( correct_label )
  df = df.rename( columns={ 'utterance': 'text'} )
  return df

combined = process( cdf )
combined.to_csv(f"{DATA_PATH}/combined.csv")

train = pd.read_json( data_paths['train.json'] )
test  = pd.read_json( data_paths['test.json'] )

train = process( train )
test  = process( test )

#split test_df into test and validation
test, val = train_test_split( test, test_size=0.5, random_state=42 )

train.to_csv(f"{DATA_PATH}/train.csv", index=False)
val.to_csv(f"{DATA_PATH}/val.csv", index=False)
test.to_csv(f"{DATA_PATH}/test.csv", index=False)