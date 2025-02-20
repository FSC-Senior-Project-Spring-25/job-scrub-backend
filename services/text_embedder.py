from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


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

        # Normalize embeddings to unit length (L2 norm)
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

    def compute_similarity(self, texts1: list[str], texts2: list[str]) -> torch.Tensor:
        """
        Compute the adjusted similarity between two sets of texts
        :return: a similarity matrix with rescaled values between -1 and 1
        """
        # Get normalized embeddings for both sets of texts
        embeddings1 = self.get_embeddings(texts1)
        embeddings2 = self.get_embeddings(texts2)

        # Calculate cosine similarity
        similarity = torch.mm(embeddings1, embeddings2.T)

        # Apply scaling to better distribute similarity scores
        # This transforms the typical BGE similarity range (0.6-0.9) to a more interpretable range
        scaled_similarity = self._rescale_similarity(similarity)

        return scaled_similarity

    @staticmethod
    def _rescale_similarity(similarity: torch.Tensor) -> torch.Tensor:
        """
        Rescale the similarity scores to better reflect semantic relationships
        """
        sim_np = similarity.numpy()

        # Define the typical range of BGE similarities for scaling
        typical_min = 0.6  # Typical minimum similarity from BGE
        typical_max = 0.9  # Typical maximum similarity from BGE

        # Rescale to [-1, 1] range
        rescaled = (sim_np - typical_min) / (typical_max - typical_min) * 2 - 1

        # Clip values to ensure they stay in [-1, 1]
        rescaled = np.clip(rescaled, -1, 1)

        return torch.from_numpy(rescaled)
