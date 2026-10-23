# Cora Citation Network — Paper Classification with a GCN

Classifies research papers in the [Cora citation network](https://graphsandnetworks.com/the-cora-dataset/)
into seven topics, using both the words in each paper and the citation links between papers.

The point of the project is a controlled comparison: **how much does the graph structure add over the
text alone?**

| Model | Sees words | Sees citations | Accuracy | Macro-F1 |
|---|---|---|---|---|
| TF-IDF + Random Forest | ✅ | ❌ | 0.581 | 0.570 |
| Two-layer GCN | ✅ | ✅ | **0.792** | **0.782** |

Both models receive identical node features. The only difference is that the GCN also receives the
edges — so the 21-point gap is attributable to the citation structure.

---

## Dataset

| | |
|---|---|
| Nodes (papers) | 2,708 |
| Edges (citations) | 5,278, stored as 10,556 directed entries |
| Features per paper | 1,433 binary bag-of-words flags |
| Classes | 7 |
| Split | 140 train / 500 validation / 1,000 test (standard Planetoid public split) |

Average degree is 3.9, so the graph is sparse; `edge_index` is stored in COO format rather than as an
adjacency matrix, which would be 99.86% zeros.

## Model

```
GCNConv(1433 → 32) → ReLU → Dropout(0.5) → GCNConv(32 → 7) → logits
```

46,119 parameters. Adam, lr 0.01, up to 200 epochs with early stopping on validation accuracy
(patience 20) and best-checkpoint restoration. The test set is evaluated once, after training.

The model returns raw logits — `F.cross_entropy` applies log-softmax internally, so softmax is applied
in the API for human-readable output rather than in the model.

## Serving

The trained model is exported to ONNX with dynamic node and edge axes, and served through FastAPI.
**Inference requires only `onnxruntime` and `numpy`** — PyTorch and PyTorch Geometric are needed to
train and to regenerate the precomputed graph, not to serve it. The full runtime is under 50MB.

| Endpoint | Description |
|---|---|
| `GET /` | Browser frontend |
| `GET /health` | Service status and execution providers |
| `GET /info` | Architecture, class map, and the ONNX input/output signatures |
| `POST /predict` | Classify an arbitrary graph (see the note under Limitations) |
| `POST /predict/cora_node` | Classify Cora nodes by index, with the true label for comparison |

```bash
curl -X POST localhost:8000/predict/cora_node \
     -H "Content-Type: application/json" \
     -d '{"node_indices": [0, 1, 2707]}'
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# open http://localhost:8000
```

To regenerate `cora_data.npz` from the raw dataset (only needed if the data changes):

```bash
pip install torch torch-geometric
python prepare_data.py
```

## Deploying

- **Render** — `render.yaml` is included; connect the repo and deploy as a web service.
- **Docker / Hugging Face Spaces / Cloud Run** — a `Dockerfile` is included, listening on `$PORT`
  (default 7860 for Spaces).

## Limitations

Stated plainly, because they bound what the numbers mean.

- **Transductive.** The full graph, including test nodes, is present during training; only the labels
  are masked. No label information leaks, but the model has never predicted on a node it did not see.
  It cannot classify a paper published tomorrow. An inductive method such as GraphSAGE would be needed
  for that.
- **Fixed vocabulary.** Features are 1,433 binary flags over a vocabulary defined by the dataset, which
  is not published with it. `POST /predict` therefore demonstrates that the ONNX export accepts dynamic
  shapes rather than serving as a general-purpose endpoint.
- **Single run, single seed.** No `torch.manual_seed` is set, so results vary between runs. Mean and
  standard deviation across several seeds would be a firmer claim than one number.
- **No weight decay.** Training accuracy reaches 1.0 by around epoch 20 on 140 labelled nodes. The
  standard GCN configuration uses `weight_decay=5e-4`; adding it is the most obvious next step.
- **Homophily assumption.** The result depends on connected papers tending to share a topic. On a
  heterophilous graph a GCN would likely underperform the feature-only baseline.

## Repository layout

```
main.py                                  FastAPI + onnxruntime service
prepare_data.py                          Regenerates cora_data.npz (needs PyTorch Geometric)
cora_data.npz                            Precomputed features, edge_index, labels, masks
simple_gcn_cora.onnx  + .onnx.data       Exported model (46,119 parameters)
static/                                  Browser frontend
streamlit_app.py                         Alternative Streamlit explorer
cora_citation_network_classification.ipynb   Training, baseline comparison and evaluation
Dockerfile  render.yaml  runtime.txt     Deployment
```
