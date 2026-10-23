"""
Cora GCN inference API.

Serves a two-layer Graph Convolutional Network, exported to ONNX, over FastAPI.
Inference depends only on onnxruntime and numpy — PyTorch and PyTorch Geometric
are needed to TRAIN and to regenerate cora_data.npz (see prepare_data.py), not to serve.
"""
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}

FEATURE_DIM = 1433

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "simple_gcn_cora.onnx")
DATA_PATH  = os.path.join(BASE_DIR, "cora_data.npz")
STATIC_DIR = os.path.join(BASE_DIR, "static")


# ---------------------------------------------------------------------------
# Startup / shutdown: load the ONNX session and the Cora arrays once.
# ---------------------------------------------------------------------------
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX model and Cora graph...")
    state["session"] = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    cora = np.load(DATA_PATH)
    state["x"] = cora["x"].astype(np.float32)
    state["edge_index"] = cora["edge_index"].astype(np.int64)
    state["y"] = cora["y"].astype(np.int64)
    print(f"Ready: {state['x'].shape[0]} nodes, {state['edge_index'].shape[1]} edges.")

    yield

    state.clear()


app = FastAPI(
    title="Cora GCN API",
    description="Node classification on the Cora citation network with a two-layer GCN.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # "*" origins and credentials are not a valid combination
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class GraphPredictRequest(BaseModel):
    node_features: List[List[float]] = Field(..., min_length=1)
    edge_indices: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_indices: List[int] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)   # numerical stability
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def run_model(node_features: np.ndarray, edge_index: np.ndarray, node_indices_to_return):
    session = state.get("session")
    if session is None:
        raise HTTPException(503, "Model is not loaded yet. Please try again shortly.")

    logits = session.run(
        ["logits"],
        {
            "node_features": node_features.astype(np.float32),
            "edge_indices": edge_index.astype(np.int64),
        },
    )[0]

    probabilities = softmax(logits)
    predicted_classes = logits.argmax(axis=-1)

    results = []
    for i in node_indices_to_return:
        results.append({
            "node_index": int(i),
            "predicted_class_id": int(predicted_classes[i]),
            "predicted_class_name": CORA_CLASSES[int(predicted_classes[i])],
            "probabilites": probabilities[i].tolist(),
            "logits": logits[i].tolist(),
        })

    return {
        "num_nodes": int(node_features.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "predictions": results,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"service": "Cora GCN API", "status": "Running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "session" in state,
        "providers": state["session"].get_providers() if "session" in state else [],
    }


@app.get("/info")
def model_info():
    session = state.get("session")
    if session is None:
        raise HTTPException(503, "Model is not loaded yet.")
    return {
        "model_name": "SimpleGCN",
        "architecture": "GCNConv(1433 -> 32) -> ReLU -> Dropout(0.5) -> GCNConv(32 -> 7)",
        "parameters": 46119,
        "feature_dimension": FEATURE_DIM,
        "num_classes": len(CORA_CLASSES),
        "class_mapping": CORA_CLASSES,
        "dataset": {
            "name": "Cora",
            "num_nodes": int(state["x"].shape[0]),
            "num_edges": int(state["edge_index"].shape[1]),
        },
        "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in session.get_outputs()],
    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    """
    Classify an arbitrary graph.

    Note: features must be 1433-dimensional over Cora's fixed vocabulary, and the
    model is transductive, so this is a demonstration that the ONNX export accepts
    dynamic node and edge counts rather than a general-purpose endpoint.
    """
    for row, feature_vector in enumerate(request.node_features):
        if len(feature_vector) != FEATURE_DIM:
            raise HTTPException(
                422, f"Row {row}: each node's feature vector must have exactly {FEATURE_DIM} features"
            )

    node_features = np.array(request.node_features, dtype=np.float32)
    num_nodes = node_features.shape[0]

    if request.edge_indices:
        edge_index = np.array(request.edge_indices, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(422, "edge_indices must have shape [2, num_edges]")
        if edge_index.size and (edge_index.max() >= num_nodes or edge_index.min() < 0):
            raise HTTPException(422, f"edge_indices must reference nodes in 0..{num_nodes - 1}")
    else:
        node_ids = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_ids, node_ids])       # self-loops only

    return run_model(node_features, edge_index, range(num_nodes))


@app.post("/predict/cora_node")
def predict_real_cora_nodes(request: CoraNodeRequest):
    """Classify nodes that exist in the Cora graph, by index."""
    x, edge_index = state.get("x"), state.get("edge_index")
    if x is None:
        raise HTTPException(503, "Dataset is not loaded yet.")

    largest_valid_index = x.shape[0] - 1
    invalid = [i for i in request.node_indices if i < 0 or i > largest_valid_index]
    if invalid:
        raise HTTPException(400, f"node_index out of bound (must be 0 to {largest_valid_index})")

    response = run_model(x, edge_index, request.node_indices)
    y = state["y"]
    for prediction in response["predictions"]:
        true_id = int(y[prediction["node_index"]])
        prediction["true_class_id"] = true_id
        prediction["true_class_name"] = CORA_CLASSES[true_id]
        prediction["correct"] = (true_id == prediction["predicted_class_id"])
    return response


# Mounted last so it does not shadow the API routes above.
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
