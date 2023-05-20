import requests, os, json
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Any, Dict
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import utils.util as helper

# Models to use for cosine similarity plots
PLOT_MODELS = (1, 3, 8)

# Filenames for the predicted and true labels
GOLD_LABEL_FILE = "sixteen_shot_true_labels.txt"
PREDICTED_LABEL_FILE = "sixteen_shot_predicted_labels.txt"

# Path to the directory where this script is located
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Path to the directory where the analysis data is located
DATA_PATH = os.path.join(CURRENT_PATH, '../../analysis')

# Initialize an empty CACHE dictionary
CACHE = {}


@dataclass
class CosineSimilarity:
    """
    A class to compute the cosine similarity between different sentences.
    """

    # API token for Hugging Face
    api_token: str

    # URL for Hugging Face's SentenceTransformer API
    API_URL: str = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        """
        Initialization function called automatically after the instance has been created.
        Initializes the sentence transformer model.
        """
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def headers(self) -> Dict[str, str]:
        """
        Returns the headers required for the Hugging Face API.
        """
        return {"Authorization": f"Bearer {self.api_token}"}

    def query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends a POST request to the Hugging Face API with the given payload.

        Args:
            payload: A dictionary containing the payload for the API.

        Returns:
            The response from the API as a dictionary.
        """
        response = requests.post(self.API_URL, headers=self.headers(), json=payload)
        return response.json()

    def get_similarity_score(self, gold_intent: str, pred_intent: str) -> float:
        """
        Gets the similarity score between the given gold intent and predicted intent.

        Args:
            gold_intent: The gold intent.
            pred_intent: The predicted intent.

        Returns:
            The similarity score as a float.
        """
        payload = {
            "inputs": {
                "source_sentence": gold_intent,
                "sentences": [pred_intent]
            }
        }
        response = self.query(payload)
        return response[0]

    def get_cosine_similarity(self, gold_intent: str, pred_intent: str) -> float:
        """
        Computes the cosine similarity between the embeddings of the given gold intent and predicted intent.

        Args:
            gold_intent: The gold intent.
            pred_intent: The predicted intent.

        Returns:
            The cosine similarity as a float.
        """
        # Compute the embedding for both intents
        embedding_gold = self.model.encode(gold_intent, convert_to_tensor=True)
        embedding_pred = self.model.encode(pred_intent, convert_to_tensor=True)

        # Compute the cosine similarity between the two embeddings
        similarity = util.pytorch_cos_sim(embedding_gold, embedding_pred).item()
        return similarity

    def compare(self, gold_intent: str, pred_intent: str) -> None:
        """
        Prints the similarity score between the given gold intent and predicted intent.

        Args:
            gold_intent: The gold intent.
            pred_intent: The predicted intent.
        """
        similarity_score = round(self.get_similarity_score(gold_intent, pred_intent) * 100, 2)
        print(f"gold: {gold_intent}\npred: {pred_intent}\nmatch: {similarity_score}%\n")

    def compare_embed(self, gold_intent: str, pred_intent: str) -> None:
        """
        Prints the cosine similarity between the given gold intent and predicted intent.

        Args:
            gold_intent: The gold intent.
            pred_intent: The predicted intent.
        """
        cosine_sim = round(self.get_cosine_similarity(gold_intent, pred_intent) * 100, 2)
        print(f"gold: {gold_intent}\npred: {pred_intent}\nmatch: {cosine_sim}%\n")

@dataclass
class MetricHandler:
    pred_file: str
    gold_file: str
    embed_handler: CosineSimilarity 

    def is_match(self, gold_intent: str, pred_intent: str) -> bool:
        """
        Checks if the gold intent matches the predicted intent exactly.

        Args:
            gold_intent: The gold intent.
            pred_intent: The predicted intent.

        Returns:
            A boolean indicating whether the gold intent matches the predicted intent exactly.
        """
        return gold_intent == pred_intent
    
    def first_match(self, gold_intent: str, pred_intent: str) -> bool:
        """
        Checks if the first word of the gold intent matches the first word of the predicted intent.

        Args:
            gold_intent: The gold intent.
            pred_intent: The predicted intent.

        Returns:
            A boolean indicating whether the first word of the gold intent matches the first word of the predicted intent.
        """
        return gold_intent.split()[0] == pred_intent.split()[0]

    def exist(self, gold_intent: str, pred_intent: str) -> bool:
        """
        Checks if both gold intent and predicted intent exist.

        Args:
            gold_intent: The gold intent.
            pred_intent: The predicted intent.

        Returns:
            A boolean indicating whether both gold intent and predicted intent exist.
        """
        return len(pred_intent) > 0 and len(gold_intent) > 0

    def calculate_accuracy(self) -> None:
        """
        Calculates the first word accuracy and exact match accuracy.

        Returns:
            A tuple containing the first word accuracy and exact match accuracy.
        """
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

    def calculate_bleu_score(self) -> float:
        """
        Calculates the BLEU score.

        Returns:
            The BLEU score.
        """
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
        """
        Calculates the Jaccard similarity between two labels.

        Args:
            label1: The first label.
            label2: The second label.

        Returns:
            The Jaccard similarity.
        """
        tokens1 = set(label1.split())
        tokens2 = set(label2.split())

        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        similarity = intersection / union if union != 0 else 0.0

        return similarity

    def calculate_jaccard_similarity(self) -> float:
        """
        Calculates the average Jaccard similarity for all the gold intents and predicted intents in the files.

        Returns:
            The average Jaccard similarity.
        """
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
        """
        Calculates the cosine similarity between two labels.

        Args:
            label1: The first label.
            label2: The second label.

        Returns:
            The cosine similarity.
        """
        return self.embed_handler.get_cosine_similarity( label1, label2 )

    def calculate_cosine_similarity(self) -> float:
        """
        Calculates the average cosine similarity for all the gold intents and predicted intents in the files.

        Returns:
            The average cosine similarity.
        """
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
        """
        Compiles the metrics into a dictionary.

        Args:
            accuracy: The first word accuracy.
            exact_match: The exact match accuracy.
            bleu_score: The BLEU score.
            jaccard_score: The Jaccard score.
            cosine_scores: The cosine similarity score.

        Returns:
            A dictionary containing all the metrics.
        """
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
        """
        Calculates all the metrics and returns them in a dictionary.

        Returns:
            A dictionary containing all the metrics.
        """
        metrics = self.get_metrics(
        accuracy = self.calculate_accuracy()[0],
        exact_match = self.calculate_accuracy()[1],
        bleu_score = self.calculate_bleu_score(),
        jaccard_score = self.calculate_jaccard_similarity(),
        cosine_scores = self.calculate_cosine_similarity()
        )
        
        return metrics
  
def load_intents_list(intents_file: str) -> list:
    """
    Load a list of intents from a file.

    Args:
        intents_file: Path to the file containing the intents.

    Returns:
        A list of intents.
    """
    with open(intents_file, 'r') as intent_preds:
        intents = [line.strip() for line in intent_preds]
    return intents


def get_cleaned_intents(gold: list, pred: list) -> list:
    """
    Clean up the predicted intents by replacing non-existent intents with 'ε'.

    Args:
        gold: List of gold intents.
        pred: List of predicted intents.

    Returns:
        A list of cleaned up predicted intents.
    """
    intent_set = set(gold)
    return ["ε" if intent not in intent_set else intent for intent in pred]


def create_metric_class(dir_path: str, label: str, pred: str) -> MetricHandler:
    """
    Evaluate the predicted intents against the gold intents and return the metrics.

    Args:
        dir_path: Path to the directory containing the intent files.
        label: Name of the file containing the gold intents.
        pred: Name of the file containing the predicted intents.

    Returns:
        A dictionary containing the evaluation metrics.
    """
    # Note: similarity_checker needs to be initialized outside of this function.
    return MetricHandler(
        embed_handler=similarity_checker,
        gold_file=f"{dir_path}/{label}",
        pred_file=f"{dir_path}/{pred}"
    )

def get_results(path: str) -> dict:
    """
    Gather evaluation results from all prediction files in a given directory.

    Args:
        path: Path to the directory containing the intent files.

    Returns:
        A dictionary containing the evaluation results for each pair of gold and prediction files.
    """
    all_results = {}
    all_files = os.listdir(path)
    labels = [file for file in all_files if "true_labels" in file]
    preds  = [file.replace("true", "predicted") for file in labels]

    for label, pred in zip(labels, preds):
        print(f"Label: {label}\nPred: {pred}")
        label_lines = open(f"{path}/{label}", "r").readlines()
        pred_lines = open(f"{path}/{pred}", "r").readlines()

        if len(label_lines) != len(pred_lines):
            print(f"{label} and {pred} are not the same length.")
            metrics = create_metric_class(path, label, pred)
            results = metrics.get_metrics(0, 0, 0, 0, 0)
        else:
            metrics = create_metric_class(path, label, pred)
            results = metrics.evaluate()

        # Update results
        results = {
            "labels_file": label,
            "preds_file": pred,
            "results": results
        } 

        all_results[f"{label.split('_')[0]}_{label.split('_')[1]}"] = results
    return all_results


def write_results(path: str, results: dict) -> None:
    """
    Write evaluation results to a JSON file.

    Args:
        path: Path to the directory where the results file should be written.
        results: A dictionary containing the evaluation results.
    """
    with open(f"{path}/results.json", "w") as f:
        json.dump(results, f, indent=4)


def get_sim_matrix(gold_intents: str, pred_intents: str) -> list:
    """
    Generate a matrix of cosine similarities between pairs of gold and predicted intents.

    Args:
        gold_intents: Path to the file containing gold intents.
        pred_intents: Path to the file containing predicted intents.

    Returns:
        A list of lists representing a matrix of cosine similarities.
    """
    # Load intents
    gold_intents = load_intents_list(gold_intents)
    pred_intents = load_intents_list(pred_intents)

    # Initialize similarity matrix
    similarity_matrix = []

    # Iterate over gold intents
    for gold in gold_intents:
        row = []
        for pred in pred_intents:
            pair = (gold, pred)

            # If pair's similarity is already calculated and in CACHE, retrieve it from the CACHE
            sim = CACHE.get(pair)
            if sim is None:
                # Else calculate similarity and store it in the CACHE
                sim = similarity_checker.get_cosine_similarity(gold, pred)
                CACHE[pair] = sim

            row.append(sim)
        similarity_matrix.append(row)

    return similarity_matrix


def get_sims(gold_intents: str, pred_intents: str) -> list:
    """
    Calculate cosine similarities for each pair of gold and predicted intents.

    Args:
        gold_intents: Path to the file containing gold intents.
        pred_intents: Path to the file containing predicted intents.

    Returns:
        A list of cosine similarities for each pair of intents, rounded to 2 decimal places.
    """
    # Load intents
    gold_intents = load_intents_list(gold_intents)
    pred_intents = load_intents_list(pred_intents)

    # Calculate cosine similarities for all pairs
    cosine_similarities = [similarity_checker.get_cosine_similarity(gold, pred) for gold, pred in zip(gold_intents, pred_intents)]
    
    # Round all values to nearest 0.01
    cosine_similarities = [round(sim, 2) for sim in cosine_similarities]

    return cosine_similarities


def create_graph(sims: list):
    """
    Create a histogram of cosine similarities.

    Args:
        sims: A list of cosine similarities.

    Returns:
        A Plotly figure object representing the histogram.
    """
    fig = go.Figure(data=[go.Histogram(x=sims, nbinsx=4, name='Cosine Similarities')])
    fig.update_layout(
        title_text='Cosine Similarity between Gold Intents and Predicted Intents', # title
        xaxis_title_text='Cosine Similarity', # x-axis title
        yaxis_title_text='Count', # y-axis title
        template='plotly_dark',
        xaxis=dict(range=[-1, 1])
    )

    # Make bars multiple colors
    fig.update_traces(marker_color=['deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'darkblue'])

    return fig

def create_heatmap(gold_intents: str, pred_intents: str, sim_mat: list) -> go.Figure:
    """
    Create a heatmap of cosine similarities between gold and predicted intents.

    Args:
        gold_intents: Path to the file containing gold intents.
        pred_intents: Path to the file containing predicted intents.
        sim_mat: Matrix of cosine similarities.

    Returns:
        A Plotly figure object representing the heatmap.
    """
    # Load intents
    gold_intents = load_intents_list(gold_intents)
    pred_intents = load_intents_list(pred_intents)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sim_mat,
        x=gold_intents,
        y=pred_intents
    ))

    # Add layout
    fig.update_layout(
        title_text='Cosine Similarity between Gold Intents and Predicted Intents',
        xaxis_title_text='Gold Intents',
        yaxis_title_text='Predicted Intents',
        template='plotly_dark',
    )

    return fig


# Initialize the cosine similarity checker
similarity_checker = CosineSimilarity(api_token="")

# Get the folders containing evaluation data
eval_folders = helper.get_folders( DATA_PATH )

# Iterate over evaluation folders
for eval_folder in eval_folders:
    models = helper.get_folders( path = f"{DATA_PATH}/{eval_folder}" )
    for model in models:

        #Get model path
        print(f"Processing {model} in {eval_folder}")
        MODEL_PATH = f"{DATA_PATH}/{eval_folder}/{model}"

        # Get and write results
        write_results( 
            path    = MODEL_PATH,
            results =  get_results( MODEL_PATH ) 
        )

        # Create cosine similarity heatmap for certain models
        if any( str(num) in model for num in PLOT_MODELS ):
            gold_path = f"{MODEL_PATH}/{GOLD_LABEL_FILE}"
            pred_path = f"{MODEL_PATH}/{PREDICTED_LABEL_FILE}"

            # Calculate similarity matrix and create heatmap
            cosine_sims = get_sim_matrix(gold_path, pred_path)
            fig = create_heatmap(gold_path, pred_path, cosine_sims)
            fig.write_image(f"{DATA_PATH}/{eval_folder}/{model}/cosine_sim.png")
