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

def create_points_for_qdrant(dataset, text_model_embeddings, image_embeddings):
    points = []
    id_counter = 1

    def get_next_id():
        nonlocal id_counter
        id_val = id_counter
        id_counter += 1
        return id_val
    
    i = 0

    for _,presentation in enumerate(dataset):
        for _,slide in enumerate(presentation["slides"]):
            points.append(
                    PointStruct(
                        id=get_next_id(), 
                        vector={
                            "all-MiniLM-L6-v2": text_model_embeddings[i],
                            client.get_sparse_vector_field_name(): encoded_sparse_docs[i],
                        },
                        payload={
                            "type": "text",
                            "file_name": presentation["file_name"],
                            "slide_number": slide["slide_number"],
                            "text": slide["text"],
                        },
                    )
                )
            i += 1
            for image_index, image_path in enumerate(slide["image_paths"]):
                points.append(
                    PointStruct(
                        id=get_next_id(),
                        vector={
                            "image-vectors": image_embeddings.pop(0),
                        },
                        payload={
                            "type": "image",
                            "file_name": presentation["file_name"],
                            "slide_number": slide["slide_number"],
                            "image_path": image_path,
                        },
                    )
                )
    return points



points = create_points_for_qdrant(dataset, text_model_embeddings, image_embeddings)


client.upload_points(
    collection_name="third Try",
    points=points
)