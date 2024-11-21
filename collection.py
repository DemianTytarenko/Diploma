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

api_key = "SvkHzA7fxPiIwvrsBq0oruGXzKxf4zIm-G8ATUJxb7GMrjvgN6QyfA"
url = "https://88e4a23d-b0b1-4348-b80d-94c77729aa8a.us-east4-0.gcp.cloud.qdrant.io:6333"


client = QdrantClient(url=url, api_key=api_key)


client.create_collection(
    "third Try",
    vectors_config={
        "all-MiniLM-L6-v2":models.VectorParams(
            size=len(text_model_embeddings[0]),
            distance=models.Distance.COSINE,
        ),
        "image-vectors": models.VectorParams(
            size=len(image_embeddings[0]),  # Размер плотного вектора для изображений
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config = client.get_fastembed_sparse_vector_params()
)