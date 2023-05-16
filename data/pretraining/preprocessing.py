
#PyTorch Imports
from transformers import T5Tokenizer, PreTrainedTokenizerBase
from datasets import ClassLabel
import torch

#Typing imports
import pandas as pd
from dataclasses import dataclass

#System Imports
import os, json, functools


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


from dataclasses import dataclass
from transformers import T5Tokenizer, PreTrainedTokenizerBase

@dataclass
class RandomNoise:
    """Randomly corrupts a span of tokens in the input."""

    tokenizer: PreTrainedTokenizerBase

    def span_corruption(self, utterances, mean_noise_span_length=3.0, noise_density=0.15):

        dataset = [self.get_random_segment(data, max_length=512) 
                   for data in list(filter(lambda x: x["targets"].shape[0] > 0, utterances))]
        
        return self.denoise(dataset, noise_density=noise_density, noise_mask_fn=functools.partial(
                self.random_spans_noise_mask,
                mean_noise_span_length=mean_noise_span_length
            )
        )

    def get_random_segment(self, data, max_length):
        """Extract a chunk from the data, given a maximum length."""
        tokens = data[ "targets" ]
        if tokens.shape[0] < max_length:
            return {"targets": tokens}
        start = torch.randint(0, tokens.shape[0] - max_length + 1, (1,)).item()
        return {"targets": tokens[start: start + max_length]}


    def random_spans_noise_mask(self, length, noise_density=0.15, mean_noise_span_length=3.0):
        """Calculate which spans to mask given input length.
        Returns a vector of Booleans of length `length`, where `True`
        corresponds to masking and `False` corresponds to keeping a token.
        """
        orig_length = length
        length = torch.tensor(length, dtype=torch.int32)
        # avoid degenerate length values
        length = torch.maximum(length, torch.tensor(2, dtype=torch.int32))
        # helper functions for concise type conversion
        def to_int(x):
            return x.type(torch.int32)
        def to_float(x):
            return x.type(torch.float32)
        # calculate number of noised and non-noised tokens
        num_noise_tokens = to_int(torch.round(to_float(length) * noise_density))
        num_noise_tokens = torch.minimum(
            torch.maximum(num_noise_tokens, torch.tensor(1, dtype=torch.int32)), length-1)
        num_noise_spans = to_int(
            torch.round(to_float(num_noise_tokens) / mean_noise_span_length))
        num_noise_spans = torch.maximum(num_noise_spans, torch.tensor(1, dtype=torch.int32))
        num_nonnoise_tokens = length - num_noise_tokens
        # pick lengths of noise spans and non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition items randomly into non-empty segments."""
            first_in_segment = torch.nn.functional.pad(
                self.shuffle(to_int(torch.arange(num_items - 1) < num_segments - 1)),
                [1, 0])
            segment_id = torch.cumsum(first_in_segment, 0)
            segment_length = self.segment_sum(torch.ones_like(segment_id), segment_id)
            return segment_length

        noise_span_lengths = _random_segmentation(
            num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = torch.reshape(
            torch.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
                        [num_noise_spans * 2])
        span_starts = torch.cumsum(interleaved_span_lengths, 0)[:-1]
        span_start_indicator = self.segment_sum(
            torch.ones_like(span_starts), span_starts, length)
        span_num = torch.cumsum(span_start_indicator, 0)
        is_noise = torch.eq(span_num % 2, torch.tensor(1, dtype=torch.int64))
        return is_noise[:orig_length]


    def denoise(self, dataset, noise_density=0.15, noise_mask_fn=None):
        vocab_size = self.tokenizer.vocab_size
        def map_fn(features):
            tokens = features['targets']
            noise_mask = noise_mask_fn(tokens.shape[0], noise_density)
            inputs = self.noise_span_to_unique_sentinel(tokens, noise_mask, vocab_size)
            return {
                'inputs': inputs,
                'targets': self.nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocab_size)
            }
        return [map_fn(data) for data in dataset]


    def noise_span_to_unique_sentinel(self, tokens, noise_mask, vocab_size):
        prev_token_is_noise = torch.nn.functional.pad(
            noise_mask[:-1], [1, 0])

        first_noise_tokens = torch.logical_and(
            noise_mask, torch.logical_not(prev_token_is_noise))
        subsequent_noise_tokens = torch.logical_and(
            noise_mask, prev_token_is_noise)

        sentinel = vocab_size - torch.cumsum(first_noise_tokens.int(), 0)

        tokens = torch.where(first_noise_tokens, sentinel, tokens)
        return torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))


    def nonnoise_span_to_unique_sentinel(self, tokens, noise_mask, vocab_size):
        return self.noise_span_to_unique_sentinel(
            tokens, torch.logical_not(noise_mask), vocab_size)


    """============= UTILITY FUNCTIONS ==============="""
    def shuffle(self, value):
        """Randomly shuffle a tensor."""
        return value[torch.randperm(value.numel())].reshape(value.shape)

    def segment_sum(self, data, segment_ids, num_segments=None):
        """Compute the sum along segments of a tensor."""
        if num_segments is None:
            num_segments = segment_ids.unique().numel()
        shape = [num_segments] + list(data.shape[1:])
        return torch.zeros(*shape, dtype=data.dtype).scatter_add_(0, segment_ids, data)

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
        if self.datahandler.args.labelsemantics in ["random_denoising", "label_denoising"]:
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
    
    def random_denoising(self):
        """
        Concatenate each intent label to its utterance and randomly noise 15% of the tokens.
        Returns preprocessed tokenized and encoded data.
        """
        utterances = [ utterance for utterance, _ in self.ic_package ]
        random_denoiser = RandomNoise( tokenizer=self.datahandler.tokenizer )
        return random_denoiser.span_corruption(
            utterances=utterances
        )
        

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

        print(f"[Preprocessing {DATASET}...]")
        for JSON in os.listdir(f"{ JSON_PATH }/{ DATASET }"):
            for pretraining_format in ["intent_classification", "label_denoising", "random_denoising"]:
                print(green(f"Preprocessing {JSON} for {pretraining_format}..."))
                args = Args(
                    dataset         = f"{ JSON_PATH }/{ DATASET }/{ JSON }",
                    seed            = 1248,
                    labelsemantics  = pretraining_format,
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

