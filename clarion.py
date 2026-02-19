from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class ClarionLite:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def chunk_text(self, text, max_tokens=900):
        tokens = self.tokenizer.encode(text, truncation=False)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = tokens[i:i+max_tokens]
            chunks.append(self.tokenizer.decode(chunk, skip_special_tokens=True))
        return chunks

    def summarize(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def hierarchical_summarize(self, document):
        chunks = self.chunk_text(document)
        intermediate_summaries = [
            self.summarize(chunk) for chunk in chunks
        ]

        combined_summary = " ".join(intermediate_summaries)
        final_summary = self.summarize(combined_summary)

        return final_summary
