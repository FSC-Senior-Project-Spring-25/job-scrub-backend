import torch
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        self.model.eval()

    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts of dimension
        :param texts: the list of texts to embed

        :return: a tensor of shape (1024)
        """
        # Tokenize and get model outputs
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get embeddings from the last hidden state
        embeddings = outputs.last_hidden_state[:, 0]
        return embeddings

    def compute_similarity(self, texts1: list[str], texts2: list[str]):
        """
        Compute the cosine similarity between two sets of texts

        :return: a similarity matrix of shape (len(texts1), len(texts2))
        """
        # Get embeddings for both sets of texts
        embeddings1 = self.get_embeddings(texts1)
        embeddings2 = self.get_embeddings(texts2)

        # Calculate cosine similarity matrix
        similarity = embeddings1 @ embeddings2.T

        return similarity
