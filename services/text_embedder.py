import torch
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        self.model.eval()

    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """ Get embeddings for a list of texts

        Args:
            texts: a list of strings to embed

        Returns:
            a torch.Tensor of dimension (1024)
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
