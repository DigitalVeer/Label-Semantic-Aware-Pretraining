from dataclasses import dataclass
from datasets import ClassLabel

import os, json
import pandas as pd

#Path settings
CURR_PATH = os.path.dirname( os.path.abspath( __file__ ) )
DATA_PATH = os.path.join( CURR_PATH, 'dataset' )

#Where to save the data
csv_cache  = f"{DATA_PATH}/csv"
json_cache = f"{DATA_PATH}/json"
preprocessed_cache = f"{CURR_PATH}/preprocessed_data"

def create_dir( path ):
    os.makedirs( path, exist_ok=True )

#Create directories
create_dir( csv_cache )
create_dir( json_cache )
create_dir( preprocessed_cache )

#Path settings
JSON_PATH  = json_cache
COMBINED_JSON_PATH = f"{ JSON_PATH }/combined"

# All data folders
data_folders = ['polyai-bank', 'wikihow']

@dataclass
class DFHandler:
    folder_name: str

    def __post_init__(self):
        self.data = pd.DataFrame()

    def add_data(self, data):
        self.data = pd.concat([self.data, data])

    def get_data(self):
        """Gets the data from the data folder."""
        curr_folder = os.path.join(CURR_PATH, self.folder_name)
        files = os.listdir(f'{curr_folder}/data')
        return {file: pd.read_csv(f'{curr_folder}/data/{file}') for file in files}
        
    def convert_intent_labels_to_integers(self, df):
        """Converts the intent labels in a DataFrame to integers.

        Args:
            df: The DataFrame to convert.

        Returns:
            The converted DataFrame.
        """

        # Drop the 'label' column.
        df = df.drop('label', axis=1)

        # Convert each intent to a ClassLabel.
        labels = df['label_name'].unique().tolist()
        ClassLabels = ClassLabel(num_classes=len(labels), names=labels)

        # Append ClassLabels into DataFrame.
        def map_label2id(row):
            return ClassLabels.str2int(row)

        df['label'] = df['label_name'].apply(map_label2id)

        # Reset the index of the DataFrame.
        df = df.reset_index(drop=True)

        return df
    
    def rename_columns(self, df):
        """Renames the columns in a DataFrame.

        Args:
            df: The DataFrame to rename.

        Returns:
            The renamed DataFrame.
        """

        df = df.rename(columns={'text': 'utterance', 'label_name': 'intent'})

        return df
    
    def write_to_json( self, df, output_file ):
        with open( output_file, 'w' ) as out_data:
            for _, row in df.iterrows():
                utterance = row["utterance"]
                intent    = row["intent"]

                json_obj = json.dumps({"translation":
                    {"src": utterance, "tgt": intent, "prefix": "intent classification: "}
                })
                out_data.write(json_obj + '\n')

    def clean_df( self, df ):
        new_df = self.convert_intent_labels_to_integers(df)
        new_df = self.rename_columns( new_df )
        return new_df


    def create_datasets( self ):
        all_data = self.get_data()
        for file, df in all_data.items():
            #Clean data
            df = self.clean_df( df )
            
            #Create naming scheme and remove ".csv"
            folder_file = f"{self.folder_name}_{file}".replace(".csv", "")

            #Check if folder exists
            if not os.path.exists( f"{csv_cache}/{self.folder_name}" ):
                os.makedirs( f"{csv_cache}/{self.folder_name}" )

            if not os.path.exists( f"{json_cache}/{self.folder_name}" ):
                os.makedirs( f"{json_cache}/{self.folder_name}" )

            #Save to csv and json
            df.to_csv( f'{csv_cache}/{self.folder_name}/{folder_file}.csv' )
            self.write_to_json( df, f'{json_cache}/{self.folder_name}/{folder_file}.json' )
            

@dataclass
class Args:
    """Arguments for pretraining"""

    dataset: str
    seed: int
    labelsemantics: str
    tokenizer: str

    def __post_init__(self):
        assert self.dataset.endswith(".json")
        assert self.labelsemantics in ["random_denoising", "intent_classification", "label_denoising"]


from transformers import T5Tokenizer, PreTrainedTokenizerBase
import torch, json

@dataclass
class DataHandler:
    """
    DataHandler class for preprocessing data for pretraining. Responsible for tokenization and writing to file.
    """

    args: Args
    punc: tuple 

    def __post_init__(self):
        self.tokenizer : PreTrainedTokenizerBase = T5Tokenizer.from_pretrained( self.args.tokenizer )

    def to_dict(self, text, include_eos=True):
        target = self.tokenizer.encode(text) if include_eos else self.tokenizer.encode(text)[:-1]
        return {'inputs': "",
                'targets': torch.tensor(target)}

    def clean_str( self, txt ):
        str = txt.strip()
        if not str.endswith( self.punc ):
            str += "."
        return str

    def load_data(self, in_file, prefix=""):
        utterances, intents = [], []
        with open( in_file, 'r') as datastrings:
            for datastring in datastrings:
                data = json.loads(datastring)

                #Clean utterance and intent
                utterance = self.clean_str( prefix + data["translation"]["src"] )
                intent    = self.clean_str( data["translation"]["tgt"] )

                #Tokenize and append to list
                utterances.append( self.to_dict(utterance, include_eos=False))
                intents.append( self.to_dict(intent, include_eos=True))

        return (utterances, intents)


    def write_data(self, dataset):
        labelsemantics = self.args.labelsemantics
        file_name      = self.args.dataset.split("/")[-1].split(".")[0]

        if not os.path.exists( f"{preprocessed_cache}/{labelsemantics}" ):
            os.makedirs( f"{preprocessed_cache}/{labelsemantics}" )

        with open( f"{preprocessed_cache}/{labelsemantics}/{file_name}.json", "w" ) as out_file:
            for data in dataset:
                data = {"inputs": self.tokenizer.decode( data["inputs"] ),
                        "targets": self.tokenizer.decode( data["targets"] )}
                out_file.write( json.dumps( data ) + "\n")


@dataclass
class Preprocessor:
    """ Preprocessor class for preprocessing data for pretraining. Responsible for implementing masking strategies. """

    datahandler: DataHandler

    def __post_init__(self):
        self.utterances, self.intents = self.datahandler.load_data( self.datahandler.args.dataset )
        self.ic_package = zip( self.utterances, self.intents )

    def label_denoise( self ):
        """Preprocessing for T5 denoising objective. Returns preprocessed
        tokenized and encoded data."""
        ds = []

        for utterance, intent in self.ic_package:
            sentinel_id = self.datahandler.tokenizer.convert_tokens_to_ids( "<extra_id_0>" )
            input = torch.cat(( utterance["targets"], torch.tensor([sentinel_id]) ))
            target = torch.cat(( torch.tensor([sentinel_id]), intent["targets"] ))
            if input.shape[0] > 512:
                input = input[:512]
            ds.append( {'inputs': input, 'targets': target} )
        return ds
    
    def intent_classification( self ):
        """Preprocessing for T5 intent classification objective. Returns preprocessed
        tokenized and encoded data."""

        ds = []
        prefix = "intent classification: "
        self.utterances, self.intents = self.datahandler.load_data( self.datahandler.args.dataset, prefix=prefix )
        self.ic_package = zip( self.utterances, self.intents )

        for utterance, intent in self.ic_package:
            input   = utterance["targets"]
            target  = intent["targets"]
            if input.shape[0] > 512:
                input = input[: 512 ]
            ds.append( {'inputs': input, 'targets': target} )
        return ds
    
    def random_denoising( self ):
        pass

    def format_pretraining( self ):
        pretrain_format = self.datahandler.args.labelsemantics
        if pretrain_format == "label_denoising":
            return self.label_denoise()
        elif pretrain_format == "intent_classification":
            return self.intent_classification()
        elif pretrain_format == "random_denoising":
            return self.random_denoising()
        else:
            raise ValueError("Invalid pretraining format")  
        


def combine_all_jsons():
    train_jsons, val_jsons = [], []

    for dataset in os.listdir( JSON_PATH ):
        for json in os.listdir(f"{ JSON_PATH }/{ dataset }"):
            
            #Get the path to the json
            curr_path = f"{ JSON_PATH }/{ dataset }/{ json }"

            #Combine train jsons
            if json.endswith("train.json"):
                train_jsons.append( curr_path )

            #Combine val jsons
            if json.endswith("val.json"):
                val_jsons.append( curr_path )

    #Create the combined jsons directory
    os.makedirs( COMBINED_JSON_PATH, exist_ok=True )

    #Create the output files
    output_files = {
        'train':    open(f"{COMBINED_JSON_PATH}/full_train.json", "w"),
        'val':      open(f"{COMBINED_JSON_PATH}/full_val.json", "w"),
        'combined': open(f"{COMBINED_JSON_PATH}/full_combined.json", "w"),
    }

    for train_json, val_json in zip( train_jsons, val_jsons ):
        for json_file, out_key in [ (train_json, 'train'), (val_json, 'val') ]:
            with open( json_file, "r" ) as in_file:
                for line in in_file:
                    output_files[ out_key ].write( line )
                    output_files[ 'combined' ].write( line )

    # Close files
    for file in output_files.values():
        file.close()

def green(text):
    return f"\033[92m\033[1m{text}\033[0m"

def highlight(text):
    return f'\x1b[6;30;42m{text}\x1b[0m'


if __name__ == "__main__":

    #Create datasets
    for folder in data_folders:
        print(green(f'Creating dataset for {folder}'))
        dh = DFHandler( folder )
        dh.create_datasets()
        print(f"Dataset created for {folder}.\nCSV Location: {csv_cache}/{folder}\nJSON Location: {json_cache}/{folder}\n")

    print(f"{highlight('All datasets created.')}\n")

    #Create a combined dataset
    combine_all_jsons()

    print(f"Beginning preprocessing...\n{'-'*50}\n")

    #Get all the jsons
    JSON_DIR   = os.listdir( JSON_PATH )

    #Begin preprocessing
    for DATASET in JSON_DIR:

        print(f"[ Preprocessing {DATASET}... ]\n")
        for JSON in os.listdir(f"{ JSON_PATH }/{ DATASET }"):
            args = Args(
                dataset         = f"{ JSON_PATH }/{ DATASET }/{ JSON }",
                seed            = 1248,
                labelsemantics  = "intent_classification",
                tokenizer       = "t5-base",
            )

            datahandler = DataHandler(
                punc = (".", "?", "!", ",", ";", ":"),
                args = args
            )

            preprocess = Preprocessor(
                datahandler = datahandler,
            )

            datahandler.write_data( preprocess.format_pretraining() )

        print(highlight(f"Finished preprocessing {DATASET}"))

    print("-"*50)

