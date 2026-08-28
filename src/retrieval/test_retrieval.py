from __future__ import annotations

import argparse 
import logging
from typing import Any

import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoTokenizer 

##Configuration

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME  = "insurance_policy_chunks_bge_m3_v1"
VECTOR_NAME = "dense"
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cpu"
MAX_LENGTH = 512
DEFAULT_TOP_K =5
LOGGER = logging.getLogger(__name__)

## Mean Pooling ##

def mean_pooling( token_embeddings: torch.Tensor, attention_mask :  torch.Tensor,) -> torch.Tensor:

    """Convert token-level embeddings into one vector per query.
    token_embeddings shape: [batch_size, sequence_length, hidden_size]
    attention_mask shape: [batch_size, sequence_length]
    output shape: [batch_size, hidden_size]
    """
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    expanded_mask = expanded_mask.to(token_embeddings.dtype)

    masked_embeddings = token_embeddings * expanded_mask 

    summed_embeddings= masked_embeddings.sum(dim=1)
    valid_token_counts = expanded_mask.sum(dim=1).clamp(min = 1e-9)

    pooled_embeddings = (summed_embeddings/ valid_token_counts)
    return pooled_embeddings

class QueryEncoder:
    """
    Encode user questions with the same BGE-M3 model used for documents.
    """

    def __init__(self, model_name : str = MODEL_NAME, 
                device : str = DEVICE,
                max_length : int =MAX_LENGTH
                ) -> None:

        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = int(max_length)

        LOGGER.info("Loading query embedding model; %s", self.model_name
                    )   

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()

        LOGGER.info("Query Embedding model loaded on %s", self.device)


    def encode(self, query : str) -> list[float]:

        """
        Encode one question into a normalized 1024 dimensional vector
        """

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty")

        encoded_input = self.tokenizer(
            query, 
            padding = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = "pt"
        )

        encoded_input = {
            key : value.to(self.device)
            for key, value in encoded_input.items()

        }

        with torch.inference_mode():
            model_output = self.model(**encoded_input)

            query_embedding = mean_pooling(
                token_embeddings = model_output.last_hidden_state,
                attention_mask = encoded_input["attention_mask"]
            )

            query_embedding =F.normalize(query_embedding,
                                         p=2,
                                         dim=1)

            if query_embedding.shape != (1,1024):

                raise RuntimeError("Unexpected query embedding shape: "
                                    f"expected=(1, 1024),"
                                    f"found={tuple(query_embedding.shape)}"
                                   )

            vector_norm = torch.linalg.vector_norm(query_embedding[0]).item()

            LOGGER.info("Query vector generated: shape=%s, norm=%.6f", tuple(query_embedding.shape), vector_norm)

            return (
                query_embedding[0].detach().cpu().to(torch.float32).tolist()
            )


# ----------------------------
# Qdrant retrieval
# ----------------------------

def retrieve_chunks(
client : QdrantClient,
query_vector : list[float],
top_k : int,
) -> list[Any]:

    """
    Retrieve the most semantically similar policy chunks.
    """

    response = client.query_points(collection_name =COLLECTION_NAME,
                                   query=query_vector,
                                   using =VECTOR_NAME,
                                   limit = top_k,
                                   with_payload = True,
                                   with_vectors=False,
                                   )

    return list(response.points)
    

# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def display_results(
    query: str,
    results: list[Any],
) -> None:
    """
    Display retrieval scores, citations and chunk text.
    """

    print("\n" + "=" * 90)
    print(f"QUERY: {query}")
    print("=" * 90)

    if not results:
        print("No matching chunks were returned.")
        return

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        source_file = payload.get("source_file", "Unknown source",)
        page_number = payload.get("page_number", "Unknown page",)
        chunk_id = payload.get("chunk_id","Unknown chunk",)
        text = payload.get("text","",)

        stripped_text = text.strip()

        print(f"\n{'=' * 100}")
        print(f"Rank: {rank}")
        print(f"Score: {result.score:.6f}")
        print(f"Point ID: {result.id}")

        print("\n--- SOURCE METADATA ---")
        print(f"Document ID: {result.payload.get('document_id')}")
        print(f"Source file: {result.payload.get('source_file')}")
        print(f"Page number: {result.payload.get('page_number')}")
        print(f"Page chunk index: {result.payload.get('page_chunk_index')}")
        print(f"Strategy: {result.payload.get('strategy')}")

        print("\n--- TEXT DIAGNOSTICS ---")
        print(f"Raw text length: {len(text)}")
        print(f"Stripped text length: {len(stripped_text)}")
        print(f"Word count: {len(stripped_text.split())}")
        print(f"Unique character count: {len(set(stripped_text))}")
        print(f"Text repr: {repr(text)}")

        print("\n--- DISPLAY TEXT ---")
        print(stripped_text)


# --------------------------------
# Arguments
# --------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first BGE-M3 and Qdrant retrieval test."
        )
    )

    parser.add_argument("--query", type=str, required=True, help="Insurance question to search for.",)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.",)

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    arguments = parse_arguments()

    if arguments.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")

    client = QdrantClient(url=QDRANT_URL, timeout=60,)

    # Fail immediately if Qdrant is not reachable.
    client.get_collection(COLLECTION_NAME)
    LOGGER.info( "Connected to collection: %s", COLLECTION_NAME,)
    encoder = QueryEncoder()
    query_vector = encoder.encode(arguments.query)
    results = retrieve_chunks( client=client, query_vector=query_vector, top_k=arguments.top_k,)
    display_results(query=arguments.query,results=results,)


if __name__ == "__main__":
    main()