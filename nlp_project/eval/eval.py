import requests, json, os
from dataclasses import dataclass
from typing import Any, Dict
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import plotly.graph_objects as go

# Initialize cache dictionary
cache = {}

#Models to plot cosine similarity
plot_models = (3, 8)
n_shot_gold = "sixteen_shot_true_labels.txt"
n_shot_pred = "sixteen_shot_predicted_labels.txt"

#Path settings
CURR_PATH = os.path.dirname( os.path.abspath( __file__ ) )
DATA_PATH = os.path.join( CURR_PATH, '../../analysis' )

@dataclass
class CosineSimilarity:
    api_token: str
    API_URL: str = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__( self ):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(self.API_URL, headers=self.headers(), json=payload)
        return response.json()

    def get_similarity_score(self, gold_intent: str, pred_intent: str) -> float:
        data = self.query(
            {
                "inputs": {
                    "source_sentence": gold_intent,
                    "sentences": [pred_intent]
                }
            })
        return data[0]
    
    def get_cosine_similarity(self, gold_intent: str, pred_intent: str) -> float:

        #Compute embedding for both lists
        embedding_1 = self.model.encode( gold_intent, convert_to_tensor=True)
        embedding_2 = self.model.encode( pred_intent, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        return sim


    def compare(self, gold_intent: str, pred_intent: str) -> None:
        cosine_sim = round(self.get_similarity_score(gold_intent, pred_intent) * 100, 2)
        print(f"gold: {gold_intent}\npred: {pred_intent}\nmatch: {cosine_sim}%\n")

    def compare_embed(self, gold_intent: str, pred_intent: str) -> None:
        cosine_sim = round(self.get_cosine_similarity(gold_intent, pred_intent) * 100, 2)
        print(f"gold: {gold_intent}\npred: {pred_intent}\nmatch: {cosine_sim}%\n")


@dataclass
class EvaluationMetricsDemo:
  pred_file: str
  gold_file: str
  embed_handler: CosineSimilarity 

  def is_match(self, gold_intent: str, pred_intent: list) -> bool:
        return gold_intent == pred_intent
    
  def first_match(self, gold_intent: str, pred_intent: list) -> bool:
     return gold_intent.split()[0] == pred_intent.split()[0]

  def exist(self, gold_intent: str, pred_intent: list) -> bool:
    return len( pred_intent ) > 0 and len( gold_intent ) > 0

  def calculate_accuracy(self) -> None:
    with open(self.pred_file, "r") as pred_f, open(self.gold_file, "r") as gold_f:
      pred_lines = pred_f.readlines()
      gold_lines = gold_f.readlines()
      
      assert len(pred_lines) == len(gold_lines)

    total: float = 0.0
    first_word_correct: float = 0.0
    exact_match: float = 0.0

    for pred_line, gold_line in zip(pred_lines, gold_lines):
        if self.gold_file.endswith("json"):
            gold_intent = json.loads(gold_line)["translation"]["tgt"]
        else:
            gold_intent = gold_line.strip()

        pred_intent = pred_line.strip()
    
        total += 1.0
        if self.first_match(gold_intent, pred_intent):
            first_word_correct += 1.0
        if self.exist( gold_intent, pred_intent ) and self.is_match( gold_intent, pred_intent ):
            exact_match += 1.0

    first_word_correct = round( first_word_correct / total * 100, 2 )
    exact_match = round( exact_match / total * 100, 2 )

    return first_word_correct, exact_match

  def calculate_bleu_score(self) -> None:
    smoothie = SmoothingFunction().method1 
    with open(self.pred_file, "r") as pred_f, open(self.gold_file, "r") as gold_f:
        pred_lines = pred_f.readlines()
        gold_lines = gold_f.readlines()

        assert len(pred_lines) == len(gold_lines)

    total: float = 0.0
    blue_scores: list = []

    for pred_line, gold_line in zip(pred_lines, gold_lines):
        if self.gold_file.endswith("json"):
            gold_intent = json.loads(gold_line)["translation"]["tgt"]
        else:
            gold_intent = gold_line.strip()
        pred_intent = pred_line.strip()

        total += 1.0
        reference = [gold_intent.split()]
        hypothesis = pred_intent.split()
        blue_scores.append(sentence_bleu(reference, hypothesis, smoothing_function=smoothie))

    blue_score = sum(blue_scores) / total * 100
    
    return blue_score
  
  def jaccard_similarity(self, label1: str, label2: str) -> float:
    # Tokenize the intent labels
    tokens1 = set(label1.split())
    tokens2 = set(label2.split())

    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    similarity = intersection / union if union != 0 else 0.0

    return similarity

  def calculate_jaccard_similarity(self) -> None:
    with open(self.pred_file, "r") as pred_f, open(self.gold_file, "r") as gold_f:
        pred_lines = pred_f.readlines()
        gold_lines = gold_f.readlines()

        assert len(pred_lines) == len(gold_lines)

    total: float = 0.0
    jaccard_scores: list = []

    for pred_line, gold_line in zip(pred_lines, gold_lines):
        if self.gold_file.endswith("json"):
            gold_intent = json.loads(gold_line)["translation"]["tgt"]
        else:
            gold_intent = gold_line.strip()
        pred_intent = pred_line.strip()

        total += 1.0
        jaccard_scores.append( self.jaccard_similarity( gold_intent, pred_intent ) )

    jaccard_score = sum(jaccard_scores) / total * 100
    
    return jaccard_score
  
  def cosine_similarity(self, label1: str, label2: str) -> float:
    return self.embed_handler.get_cosine_similarity( label1, label2 )

  def calculate_cosine_similarity(self) -> None:
    with open(self.pred_file, "r") as pred_f, open(self.gold_file, "r") as gold_f:
        pred_lines = pred_f.readlines()
        gold_lines = gold_f.readlines()

        assert len(pred_lines) == len(gold_lines)

    cosine_scores: int = 0
    total: float = 0.0

    for pred_line, gold_line in zip(pred_lines, gold_lines):
        if self.gold_file.endswith("json"):
            gold_intent = json.loads(gold_line)["translation"]["tgt"]
        else:
            gold_intent = gold_line.strip()
        pred_intent = pred_line.strip()
        total += 1.0

        score = self.cosine_similarity( gold_intent, pred_intent )
        if score >= 0.70:
            cosine_scores += 1

    return cosine_scores / total
    
  def get_metrics(self, accuracy: float, exact_match: float, bleu_score: float, jaccard_score: float, cosine_scores: float) -> dict:
    return {
        'accuracy': {
            'first_word': accuracy,
            'exact_match': exact_match
        },
        'bleu_score': bleu_score,
        'jaccard_score': jaccard_score,
        'cosine_similarity': cosine_scores
    }

  def evaluate(self) -> dict:

    metrics = self.get_metrics(
      accuracy = self.calculate_accuracy()[0],
      exact_match = self.calculate_accuracy()[1],
      bleu_score = self.calculate_bleu_score(),
      jaccard_score = self.calculate_jaccard_similarity(),
      cosine_scores = self.calculate_cosine_similarity()
    )
    
    return metrics
  

def load_intents_list(intents_file):
    intents = []
    with open(intents_file, 'r') as intent_preds:
        for line in intent_preds:
            intents.append(line.strip())

def get_cleaned_intents(gold: list, pred: list) -> list:
    intent_set = set(gold)
    pred_intents = pred.copy()
    for idx, pred in enumerate(pred):
        if pred not in intent_set:
            pred_intents[idx] = "ε"
    return pred_intents


def evaluate_pred( dir_path: str, label: str, pred: str ) -> dict:
    metrics = EvaluationMetricsDemo(
            embed_handler=similarity_checker,
            gold_file=f"{dir_path}/{label}",
            pred_file=f"{dir_path}/{pred}"
    )
    return metrics

def get_results( path: str ) -> dict:

    all_results = {}
    all_files = os.listdir( path )
    labels = [ file for file in all_files if "true_labels" in file ]
    preds  = [ file.replace( "true", "predicted" ) for file in labels ]
    for label, pred in zip( labels, preds ):

        print( f"Label: {label}\nPred: {pred}")
        label_lines = open( f"{path}/{label}", "r" ).readlines()
        pred_lines = open( f"{path}/{pred}", "r" ).readlines()

        #If empty file
        if len( label_lines ) != len( pred_lines ):
            print(f"{label} and {pred} are not the same length.")
            results = evaluate_pred( path, label, pred ).get_metrics( 0, 0, 0, 0, 0 )
        else:
            results = evaluate_pred( path, label, pred ).evaluate()

        #Update results
        results = {
            "labels_file": label,
            "preds_file": pred,
            "results": results
        } 

        #Get few shot specificalition
        all_results[ f"{label.split('_')[0]}_{label.split('_')[1]}" ] = results
    return all_results

def write_results( path: str, results: dict ) -> None:
    json_d = { result: metrics 
        for result, metrics in results.items() }
    
    with open( f"{path}/results.json", "w" ) as f:
        json.dump( json_d, f, indent=4 )


def get_sim_matrix(gold_intents, pred_intents):
    # list of pairs of intents
    gold_intents = load_intents_list(gold_intents)
    pred_intents = load_intents_list(pred_intents)

    similarity_matrix = []
    for gold in gold_intents:
        row = []
        for pred in pred_intents:
            pair = (gold, pred)

            # If pair's similarity is already calculated and in cache, retrieve it from the cache
            if pair in cache:
                sim = cache[pair]
            else:
                # Else calculate similarity and store it in the cache
                sim = similarity_checker.get_cosine_similarity(gold, pred)
                cache[pair] = sim

            row.append(sim)
        similarity_matrix.append(row)

    return similarity_matrix


def get_sims( gold_intents, pred_intents):
    # list of pairs of intents
    gold_intents = load_intents_list(gold_intents)
    pred_intents = load_intents_list(pred_intents)

    # calculate cosine similarities for all pairs
    cosine_similarities = [similarity_checker.get_cosine_similarity(gold, pred) for gold, pred in zip(gold_intents, pred_intents)]
    #round all values to nearest 0.01
    cosine_similarities = [round(sim, 2) for sim in cosine_similarities]

    return cosine_similarities

def create_graph( sims ):
    fig = go.Figure(data=[go.Histogram(x=sims, nbinsx=4, name='Cosine Similarities')])
    fig.update_layout(
        title_text='Cosine Similarity between Gold Intents and Predicted Intents', # title
        xaxis_title_text='Cosine Similarity', # x-axis title
        yaxis_title_text='Count', # y-axis title
        template='plotly_dark',
        xaxis = dict(range=[-1, 1])
    )

    #make bars multiple colors
    fig.update_traces(marker_color=['deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'darkblue'])

    return fig


def create_heatmap( gold_intents, pred_intents, sim_mat ):
    #Read in intents
    gold_intents = load_intents_list(gold_intents)
    pred_intents = load_intents_list(pred_intents)

    # create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sim_mat,
        x=gold_intents,
        y=pred_intents
    ))

    # add layout
    fig.update_layout(
        title_text='Cosine Similarity between Gold Intents and Predicted Intents',
        xaxis_title_text='Gold Intents',
        yaxis_title_text='Predicted Intents',
        template='plotly_dark',
    )

    return fig

#Cosine Similarity Example
similarity_checker = CosineSimilarity( api_token="" )
eval_folders = os.listdir( DATA_PATH )
for eval_folder in eval_folders:
    models = os.listdir( f"{DATA_PATH}/{eval_folder}" )
    for model in models:
        print(f"Processing {model} in {eval_folder}")
        results = get_results( f"{DATA_PATH}/{eval_folder}/{model}" )
        write_results( f"{DATA_PATH}/{eval_folder}/{model}", results )
        
        #Cosine Similarity Graph if folder number contains a number from plot_models
        if any( num in eval_folder for num in plot_models ):

            gold_path = f"{DATA_PATH}/{eval_folder}/{model}/{n_shot_gold}"
            pred_path = f"{DATA_PATH}/{eval_folder}/{model}/{n_shot_pred}"

            cosine_sims = get_sim_matrix( gold_path, pred_path )
            fig = create_heatmap( gold_path, pred_path, cosine_sims )
            fig.write_image(f"{DATA_PATH}/{eval_folder}/{model}/cosine_sim.png")