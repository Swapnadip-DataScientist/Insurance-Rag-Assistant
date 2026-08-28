from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from FlagEmbedding import BGEM3FlagModel

@dataclass (frozen = True)
class EmbeddingConfig:

    #Configurations for BGE-M3 dense embedding generations
    model_name : str = "BAAI/bge-m3"

    #Current machine configuration
    device : str = "cpu"
    use_fp16 : bool =False

    #Conservative CPU batch size
    batch_size : int =4

    #Max no of token for chunk
    max_length = 512

    #Normalize vectors for Cosine Similarity
    normalized_embeddings : bool = True

    #BGE-M3 dense vector dimension
    expected_dimension : int = 1024


class BgeM3DenseEmbedder:

    """
    Dense embedding generator using BAAI/bge-m3.
    Baseline configuration:
        Dense   = enabled
        Sparse  = disabled
        ColBERT = disabled
    """

    def __init__(self, config : EmbeddingConfig) -> None:

        self.config = config

        print(f"Loading model{self.config.model_name}")
        print(f"Device{self.config.device}")


        self.model = BGEM3FlagModel( self.config.model_name, 
                                     devices = self.config.device,
                                     use_fp16 =self.config.use_fp16,
                                     pooling_method = "cls",
                                     normalize_embeddings=(self.config.normalized_embeddings),
                                     batch_size = self.config.batch_size,
                                     passage_max_length = self.config.max_length,
                                     return_dense = True,
                                     return_sparse = False,
                                     return_colbert_vecs = False,)

        print("BGE-M3 model loaded successfully")



    def encode(self, texts : list[str]) -> np.ndarray:
        """
        Generate dense embeddings.
        Input: list[str]
        Output: NumPy array with shape:[number_of_texts, 1024]
        """

        if not texts:
            raise ValueError("No text supplied for embedding")

        for index, text in enumerate(texts):

            if not isinstance(text, str):
                raise TypeError(f"Text at Index {index} must be a string")

            if not text.strip():
                raise ValueError(f"Text at index {index} is empty.")

            output =  self.model.encode_corpus(
                texts,
                batch_size = self.config.batch_size,
                max_length = self.config.max_length,
                return_dense = True,
                return_sparse = True,
                return_colbert_vecs =False,
            )

            embeddings = np.asarray(output["dense_vecs"], dtype= np.float32)

            if embeddings.ndim ==1:
                embeddings= embeddings.reshape(1,-1)

            return embeddings








