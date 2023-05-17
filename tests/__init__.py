import json, torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_dataset(dataset):
    """
    Load .json dataset.
    """
    ds = []
    with open(dataset, 'r') as in_data:
        for example in in_data:
            json_data = json.loads(example)
            ds.append(json_data)
    return ds


def calculate_metric(dataset, metric, tokenizer, model=None):
    """Calculate complexity of evaluation set. Returns float or int.
    Arguments:
        dataset: dev or test file on which we evaluate
        metric: quantitative metric of dataset complexity. Options:
            token vocabulary: number of tokens needed to represent all intents in the eval set when using
                              `tokenizer`.
    """

    if metric not in ("token vocabulary", "perplexity"):
        raise ValueError("Unrecognized metric.")

    ds = load_dataset(dataset)
    if metric == "token vocabulary":
        intent_set = set([example["translation"]["tgt"] for example in ds])
        num_intents = len(intent_set)
        print("Num intents: {}".format(num_intents))

        all_tokens = []
        for intent in intent_set:
            all_tokens.extend(tokenizer.encode(intent))
        num_unique_tokens = len(set(all_tokens))
        return num_unique_tokens

    if metric == "perplexity":
        total_log_likelihood = 0
        total_tokens = 0

        for example in ds:
            sentence = example["translation"]["tgt"]
            input_ids = tokenizer.encode(sentence, return_tensors='pt')
            
            # Calculate the log likelihood of the sentence
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                log_likelihood = outputs[0]

            total_log_likelihood += log_likelihood.item()
            total_tokens += len(input_ids[0])
            
        perplexity = torch.exp(-total_log_likelihood / total_tokens)

        return perplexity.item()


    return None


if __name__ == "__main__":

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    metric = "token vocabulary"
    dataset = "../data/evaluation/dataset/json/SNIPS/SNIPS_combined.json"
    model = T5ForConditionalGeneration.from_pretrained('t5-base').eval()

    print("{}: {}".format("perplexity", calculate_metric(dataset, metric, tokenizer, model)))