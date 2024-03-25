import torch
import torch.nn as  nn

from src.params import Parameters
from src.vocab import Vocab

class Model(nn.Module):

    def __init__(self, vocab: Vocab, params: Parameters):
        super().__init__()
        self.vocab = vocab
        self.params = params
        self.context_embed_matrix = nn.Embedding(self.vocab.get_lenght(), self.params.EMBED_DIM, max_norm = self.params.EMBED_MAX_NORM, padding_idx=self.params.SPECIALS.index("[PAD]"))
        self.target_embed_matrix = nn.Embedding(self.vocab.get_lenght(), self.params.EMBED_DIM, max_norm = self.params.EMBED_MAX_NORM, padding_idx=self.params.SPECIALS.index("[PAD]"))

    def forward(self, contexts, targets):
        context_embeddings = self.context_embed_matrix(contexts)
        target_embeddings = self.target_embed_matrix(targets)

        if self.params.TOKENIZER != "WordTokenizer":
            context_embeddings = context_embeddings.sum(dim=2)
            target_embeddings = target_embeddings.sum(dim=2)

        context_embeddings = context_embeddings.mean(dim=1).view(context_embeddings.shape[0], 1, context_embeddings.shape[2])
        target_embeddings = target_embeddings.permute(0, 2, 1)

        similarity_matrix = torch.bmm(context_embeddings, target_embeddings)
        return similarity_matrix.squeeze()
