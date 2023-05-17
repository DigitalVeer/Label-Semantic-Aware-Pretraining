from dataclasses import dataclass
import os, json
import pandas as pd

def green(text):
    return f"\033[92m\033[1m{text}\033[0m"

def highlight(text):
    return f'\x1b[6;30;42m{text}\x1b[0m'


#Path settings
CURR_PATH = ""
DATA_PATH = os.path.join( CURR_PATH, 'dataset' )

#Where to save the data
csv_cache  = f"{DATA_PATH}/csv"
json_cache = f"{DATA_PATH}/json"

def create_dir( path ):
    os.makedirs( path, exist_ok=True )

#Create directories
create_dir( csv_cache )
create_dir( json_cache )

#Path settings
JSON_PATH  = json_cache
COMBINED_JSON_PATH = f"{ JSON_PATH }/combined"

# All data folders
data_folders = ['ATIS', 'SNIPS', 'TOPS_Reminder', 'TOPS_Weather']

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
        
    def clean_data(self, df):
        """Cleans the data."""

        if 'utterance' in df.columns:
            df.rename(columns={'utterance': 'text'}, inplace=True)

        if 'label' in df.columns:
            df.rename(columns={'label': 'intent'}, inplace=True)

        df['text'] = df['text'].str.strip()
        df['intent'] = df['intent'].str.strip()

        return df
    
    def write_to_json( self, df, output_file ):
        with open( output_file, 'w' ) as out_data:
            for _, row in df.iterrows():
                utterance = row["text"]
                intent    = row["intent"]

                json_obj = json.dumps({"translation":
                    {"src": utterance, "tgt": intent, "prefix": "intent classification: "}
                })
                out_data.write(json_obj + '\n')

    def create_datasets( self ):
        all_data = self.get_data()
        for file, df in all_data.items():
                       
            #Clean data
            df = self.clean_data( df )

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


if __name__ == '__main__':
    for folder in data_folders:
        print(green(f'Creating dataset for {folder}'))
        dh = DFHandler( folder )
        dh.create_datasets()
        print(f"Dataset created for {folder}.\nCSV Location: {csv_cache}/{folder}\nJSON Location: {json_cache}/{folder}\n")
    print(highlight("All datasets created!"))