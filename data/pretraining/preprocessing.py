from dataclasses import dataclass
import os, json, pandas

JSON_PATH  = "dataset/json"
JSON_DIR   = os.listdir( JSON_PATH )

COMBINED_JSON_PATH = f"{JSON_PATH }/combined"


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

    def load_data(self, in_file):
        utterances, intents = [], []
        with open( in_file, 'r') as datastrings:
            for datastring in datastrings:
                data = json.loads(datastring)

                #Clean utterance and intent
                utterance = self.clean_str( data["translation"]["src"] )
                intent    = self.clean_str( data["translation"]["tgt"] )

                #Tokenize and append to list
                utterances.append( self.to_dict(utterance, include_eos=False))
                intents.append( self.to_dict(intent, include_eos=True))

        return (utterances, intents)


    def write_data(self, dataset):
        labelsemantics = self.args.labelsemantics
        file_name      = self.args.dataset.split("/")[-1].split(".")[0]

        if not os.path.exists( f"preprocessed_data/{labelsemantics}" ):
            os.makedirs( f"preprocessed_data/{labelsemantics}" )

        with open( f"preprocessed_data/{labelsemantics}/{file_name}.json", "w" ) as out_file:
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

        for utterance, intent in self.ic_package:
            input   = utterance["targets"]
            target  = intent["targets"]
            if input.shape[0] > 512:
                input = input[: 512 - len( prefix )]
            ds.append( {'inputs': prefix + input, 'targets': target} )
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

    for dataset in JSON_DIR:
        for json in os.listdir(f"{ JSON_PATH }/{ dataset }"):
            if json.endswith("train.json"):
                train_jsons.append(f"{ JSON_PATH }/{ dataset }/{ json }")
            elif json.endswith("val.json"):
                val_jsons.append(f"{ JSON_PATH }/{ dataset }/{ json }")

    os.makedirs( COMBINED_JSON_PATH, exist_ok=True )

    output_files = {
        'train': open(f"{COMBINED_JSON_PATH}/full_train.json", "w"),
        'val': open(f"{COMBINED_JSON_PATH}/full_val.json", "w"),
        'combined': open(f"{COMBINED_JSON_PATH}/combined.json", "w"),
    }

    for train_json, val_json in zip(train_jsons, val_jsons):
        for json_file, out_key in [(train_json, 'train'), (val_json, 'val')]:
            with open(json_file, "r") as in_file:
                for line in in_file:
                    output_files[out_key].write(line)
                    output_files['combined'].write(line)

    # Don't forget to close your files when you're done
    for file in output_files.values():
        file.close()


                

#Create a combined dataset
combine_all_jsons()

#Begin preprocessing
for DATASET in JSON_DIR:
    for JSON in os.listdir(f"{ JSON_PATH }/{ DATASET }"):
        args = Args(
            dataset         = f"dataset/json/{ DATASET }/{ JSON }",
            seed            = 1248,
            labelsemantics  = "label_denoising",
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

