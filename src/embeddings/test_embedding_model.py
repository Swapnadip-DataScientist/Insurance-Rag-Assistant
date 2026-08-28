import numpy as np

from embedding_generator import(EmbeddingConfig, BgeM3DenseEmbedder)

config = EmbeddingConfig()
embedder = BgeM3DenseEmbedder(config=config)

texts = [( "Motor insurance covers accidental "
        "damage to the insured vehicle."),
        ( "The policy excludes losses caused "
            "by normal wear and tear."),
        ]

embedding = embedder.encode( texts)

print ("Embedding is completed")
print(f"no_of_texts" , len(texts))
print("Embedding shape" ,embedding.shape )
print(embedding.dtype)
print(np.isnan(embedding).any())
print(np.isinf(embedding).any())
print("Vector norms:",np.linalg.norm(embedding,axis=1,))






