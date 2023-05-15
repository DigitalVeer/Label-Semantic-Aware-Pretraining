from datasets import load_dataset, concatenate_datasets
import numpy, pandas, os

#Set train, val, and test split sizes
TRAIN_SPLIT = 0.6
VAL_SPLIT   = 0.2
TEST_SPLIT  = 0.2

#Add Label Names for each label
def add_labelname( row ):
  return banking_dataset["train"].features["label"].int2str( row )

def correct_label( row ):
  return " ".join( [word.capitalize() for word in row.split("_")] ) + "."


#Load PolyBanking Dataset from HuggingFace
#get path to current directory
path = os.path.dirname( os.path.realpath(__file__) )
banking_dataset = load_dataset( "PolyAI/banking77", cache_dir="D:\digit\Documents\Development\HuggingFace\Datasets\PolyAIBanking")

#Return Dataframe to perform operations on
banking_dataset.set_format( type="pandas" )

train_df = banking_dataset["train"].data.to_pandas()
val_df   = banking_dataset["test"].data.to_pandas()

train_df["label_name"] = train_df["label"].apply( add_labelname )
train_df["label_name"] = train_df["label_name"].apply( correct_label )

val_df["label_name"] = val_df["label"].apply( add_labelname )
val_df["label_name"] = val_df["label_name"].apply( correct_label )

train_df.to_csv( "data/train.csv" )
val_df.to_csv( "data/val.csv" )

#Concatenate train/test into one dataframe
banking_dict = concatenate_datasets( [banking_dataset["train"], banking_dataset["test"]] )

#Extract data from dictionary
banking_df = banking_dict.data.to_pandas()

#Add label name associated with each label
banking_df["label_name"] = banking_df["label"].apply( add_labelname )
banking_df["label_name"] = banking_df["label_name"].apply( correct_label )

banking_df.to_csv( "data/combined.csv" )
