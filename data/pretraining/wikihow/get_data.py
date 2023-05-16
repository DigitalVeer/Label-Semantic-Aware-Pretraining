import numpy as np
import pandas as pd
import re

#Set train split
TRAIN_SPLIT = 0.6

def create_label_name(df):
    """
    Create a column with each sentence's intent label.
    """
    df[ 'label_name' ] = df.apply( lambda row: row[ f'ending{row["label"]}' ] + ".", axis=1 )
    return df

def drop_ending_columns(df):
    """
    Drop the 'ending' columns from the dataframe.
    """
    endings_to_drop = [ f'ending{i}' for i in range( 4 ) ]
    df = df.drop( endings_to_drop, axis=1 )
    return df

def remove_how_to(df):
    """
    Remove "How to" from utterances.
    """
    pattern = re.compile( r'(?i)How to ' )
    df[ 'label_name' ] = df[ 'label_name' ].apply( lambda x: re.sub( pattern, '', x ).strip() )
    return df

# Load data
wikihow_df = pd.read_csv( 'en_wikihow_train.csv', index_col=0 )

# Drop unnecessary columns
wikihow_df = wikihow_df.drop(['startphrase', 'video-id', 'gold-source', 'fold-ind', 'sent1'], axis = 1)

# Append label name to each column
wikihow_df = create_label_name( wikihow_df )

# Drop the 'ending' columns
wikihow_df = drop_ending_columns( wikihow_df )

# Remove "How to" from the 'label_name' column (case-insensitive)
wikihow_df = remove_how_to( wikihow_df ).rename( columns={'sent2': 'text'} )[ ['text', 'label', 'label_name'] ]

#Change label value to position i.e. 0, 1, 2, 3, 4... n (n = number of labels)
wikihow_df[ 'label' ] = wikihow_df.index.values


#Split data into train, validation and test sets
shuffled_df = wikihow_df.sample( frac = 1, random_state=42 )

#Split data into train, validation and test sets
n = len( shuffled_df )
train_size = int( TRAIN_SPLIT * n )
train, validate = np.split( shuffled_df, [ train_size ])


wikihow_df.to_csv( "data/combined.csv" )
train.to_csv( "data/train.csv" ); 
validate.to_csv( "data/val.csv" );