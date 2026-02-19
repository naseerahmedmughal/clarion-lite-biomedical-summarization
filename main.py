from datasets import load_dataset
from clarion import ClarionLite
from evaluate import compute_rouge

def run_experiment():
    dataset = load_dataset("ccdv/pubmed-summarization", split="test[:5]")
    
    model = ClarionLite()

    predictions = []
    references = []

    for sample in dataset:
        document = sample["article"]
        reference = sample["abstract"]

        summary = model.hierarchical_summarize(document)

        predictions.append(summary)
        references.append(reference)

    scores = compute_rouge(predictions, references)

    print("Evaluation Results:")
    print(scores)

if __name__ == "__main__":
    run_experiment()
