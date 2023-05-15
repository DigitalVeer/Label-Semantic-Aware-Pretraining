import os, json, dataclasses
import pandas as pd
from datasets import ClassLabel

# set the data folders
data_folders = ['polyai-bank', 'wikihow']

csv_cache  = "dataset/csv"
json_cache = "dataset/json"

@dataclasses.dataclass
class DataHandler:
    folder_name: str

    def __post_init__(self):
        self.data = pd.DataFrame()

    def add_data(self, data):
        self.data = pd.concat([self.data, data])

    def get_data(self):
        files = os.listdir(f'{self.folder_name}/data')
        return {file: pd.read_csv(f'{self.folder_name}/data/{file}') for file in files}
        
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
            
    
if __name__ == "__main__":

    dh_list = []

    for folder in data_folders:
        print("Creating dataset for", folder)
        dh = DataHandler( folder )
        dh.create_datasets()

        dh_list.append( dh )
        print(f"Dataset created for {folder}.\nCSV Location: {csv_cache}/{folder}\nJSON Location: {json_cache}/{folder}\n")

    print("All datasets created.")