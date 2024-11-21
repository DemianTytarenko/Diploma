from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from fastembed.embedding import TextEmbedding
from qdrant_client.models import Filter
from fastembed import SparseTextEmbedding
import tqdm
from fastembed import ImageEmbedding
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import os
from PIL import Image
import numpy as np
import os
import json
import uuid

text_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
image_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
client.set_sparse_model("Qdrant/bm42-all-minilm-l6-v2-attentions")

text_model_embeddings = list(text_model.embed(texts))
image_embeddings = list(image_model.embed(images))

batch_size = 16
encoded_sparse_docs = list(client._sparse_embed_documents(    documents=texts,
    embedding_model_name=client.sparse_embedding_model_name,    batch_size=batch_size,
    parallel=None,))
