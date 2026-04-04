# WorldLeaf

Hand-curated African Savanna knowledge graph with 1,423 species nodes and 1,541 ecological edges, built to support an interpretable reinforcement learning agent for multi-hop ecological reasoning. Master's thesis project.

---


![Projext Flow](Project_Flow.png)

## Overview

WorldLeaf constructs a structured knowledge graph of the Serengeti/African Savanna ecosystem and trains an RL agent to navigate it for multi-hop question answering. The project is motivated by the limitations of flat retrieval systems (RAG) on relational reasoning tasks, the RL agent is designed to exploit graph structure for interpretable, path-based answers.

The project has three stages:

- Stage 1: Dataset Construction (complete)
- Stage 2: TransE Embedding Pre-training (upcoming)
- Stage 3: RL Traversal Agent Training and Evaluation (upcoming)

---

## Knowledge Graph

- 1,423 nodes: 760 iNaturalist species + 663 parent taxonomy nodes from Wikidata
- 1,541 edges across 9 relation types: eats, preys_on, parasitizes, pollinates, parent_taxon, scavenges_from, migrates_with, disperses_seeds_of, symbiotic_with
- Node attributes: node_id (Wikidata QID), name (scientific), common_name, rank, iconic_taxon
- Stored as a NetworkX DiGraph (graph.gpickle) with CSV exports and an interactive Pyvis visualization

---

## Data Sources

- iNaturalist: Species list for Serengeti National Park (place ID 69054)
- Wikidata SPARQL:QID resolution, eats and parent_taxon edges
- GloBI (Global Biotic Interactions): Preys_on, parasitizes, pollinates edges
- Manual annotation: Scavenges_from, migrates_with, disperses_seeds_of, symbiotic_with edges
- Wikipedia: Full article text per node for RAG baseline

---

## Project Structure

```

WORLDLEAF/
├── src/dataset/
│   ├── wikidata_fetch.py        # iNaturalist species collection and Wikidata edge fetching
│   ├── globi_fetch.py           # GloBI interaction edge fetching
│   ├── graph_builder.py         # NetworkX DiGraph construction
│   ├── edge_features.py         # Edge feature computation
│   └── query_builder.py         # SPARQL query utilities
├── baseline/
│   ├── fetch_wikipedia.py       # Wikipedia article fetching via Wikidata QIDs
│   ├── embed_nodes.py           # Wikipedia embeddings (text-embedding-3-large)
│   ├── embed_edges.py           # Graph relationship embeddings
│   └── save_edge_texts.py       # Save edge texts to JSON
├── RAG/
│   ├── retriever.py             # Cosine similarity retrieval over 6,144d index
│   └── rag_pipeline.py          # End-to-end RAG answer generation
├── QA/
│   ├── generate_single_hop.py   # Single-hop QA pair generation from edges
│   ├── build_chains.py          # Multi-hop chain enumeration
│   ├── generate_multi_hop.py    # Natural language question generation via GPT-4o mini
│   └── fix_qa_dataset.py        # Group all valid answers per question
├── Evaluation/
│   └── evaluate_rag.py          # LLM-judged evaluation with failure mode breakdown
├── proposed_solution/
│   ├── edge_embeddings2/
│   │   └── embed_edges.py       # OpenAI edge pair embeddings (3,072d)
│   └── RL_agent/
│       ├── embed_queries.py     # Pre-compute query embeddings
│       ├── utils.py             # Central data loader
│       ├── environment.py       # Graph traversal, action space, reward
│       ├── policy.py            # MLP policy network
│       ├── train.py             # REINFORCE training loop
│       ├── evaluate.py          # Simple Hit@1 evaluation
│       ├── evaluate_llm.py      # GPT judge evaluation
│       └── demo.py              # End-to-end demo on sample questions
├── data/
│   ├── raw/                     # Raw edge and species files, not committed to git
│   └── processed/               # Final graph files and visualization
├── notebooks/
│   └── 01_explore_graph.ipynb   # Graph exploration notebook
├── misc/                        # Ad-hoc scripting folder
├── visualize_graph.py           # Graph visualization
└── requirements.txt


```

---

## RAG Baseline

Each node is represented by two 3,072-dimensional embeddings, one from its Wikipedia text and one from a structured text of its graph relationships, both produced using OpenAI text-embedding-3-large. At retrieval time these are concatenated into a 6,144-dimensional vector. A query is embedded, compared via cosine similarity against all 1,423 nodes, and the top-3 nodes are retrieved. Their full text is passed as context to GPT-4o mini, which answers using only the provided context.

Keeping the two embeddings separate enables ablation studies at Stage 3.

## RL Agent

The RL agent is a policy network (MLP) trained via REINFORCE to navigate the knowledge graph by following semantically relevant edges toward answer nodes. At each node the agent scores all outgoing edges and a STOP action using the policy MLP, selects the top actions via beam search, and either follows an edge or declares the current node as its answer.

### RL Agent vs RAG Baseline (LLM-judged)

## Final Test Results vs RAG Baseline

| Metric | RAG Baseline | RL Agent | Delta |
|--------|-------------|----------|-------|
| **Overall Hit@1** | 49.12% | **61.76%** | **+12.64%** |
| Single-hop Hit@1 | 89.60% | 71.60% | -18.00% |
| Multi-hop Hit@1 | 38.16% | **45.63%** | **+7.47%** |

---

## Error Analysis

| Error Type | RAG Baseline | RL Agent | Delta |
|------------|-------------|----------|-------|
| HIT | 49.12% | **61.76%** | +12.64% |
| TYPE1 (traversal miss) | 10.50% | **5.51%** | -4.99% |
| TYPE2 (data gap) | 40.30% | **31.99%** | -8.31% |
| TYPE3 (wrong info) | 0.10% | 0.74% | +0.64% |


TYPE1: Agent went to completely wrong nodes.
TYPE2: Agent found relevant nodes but missing the specific relationship info.
TYPE3: Agent found relevant nodes but info was wrong.


The RL agent improves overall performance by +12.64% and multi-hop by +7.47% by actively traversing the graph structure rather than relying on flat embedding similarity. RAG retains an advantage on single-hop questions where direct retrieval is sufficient.

---

## QA Dataset

- The dataset is available [here](https://drive.google.com/drive/folders/17X2n96Rr8LDyVdhGqN1Vz80HRhRMbsL7?usp=sharing).
- 2,276 single-hop QA pairs generated from all edges using typed templates (forward and backward directions)
- 8,405 multi-hop QA pairs generated from all valid 2-hop and 3-hop chains
- Chain validity rules: no circular paths, max one parent_taxon per chain, parent_taxon cannot be first relation, must include at least one ecological relation, no same relation twice in a row
- Questions generated via GPT-4o mini with ecology expert prompting
- Total: 10,681 QA pairs

---

## Setup

```bash
conda create -n worldleaf python=3.10
conda activate worldleaf
pip install -r requirements.txt
```
Set your OpenAI API key as an environment variable:



---

## Reproducing the Pipeline

Since large data files are not included in this repository, follow these steps to reproduce:

**Stage 1 — Data Generation:**

1. Collect species and build graph: `python src/dataset/wikidata_fetch.py`
2. Fetch GloBI edges: `python src/dataset/globi_fetch.py`
3. Build graph: `python src/dataset/graph_builder.py`
4. Fetch Wikipedia text: `python Baseline/fetch_wikipedia.py`
5. Embed Wikipedia text: `python Baseline/embed_nodes.py`
6. Embed edge relationships: `python Baseline/embed_edges.py`
7. Save edge texts: `python Baseline/save_edge_texts.py`
8. Generate single-hop QA: `python QA/generate_single_hop.py`
9. Build multi-hop chains: `python QA/build_chains.py`
10. Generate multi-hop questions: `python QA/generate_multi_hop.py`

**Stage 2 — Implement Baseline:**

11. Run RAG evaluation: `python Evaluation/evaluate_rag.py`

**Stage 3 — Edge Embeddings:**

13. Embed edge pairs: `python proposed_solution/edge_embeddings2/embed_edges.py`

**Stage 4 — RL Agent:**

14. Pre-compute query embeddings: `python proposed_solution/RL_agent/embed_queries.py`
15. Train RL agent: `python proposed_solution/RL_agent/train.py`
16. Run simple evaluation: `python proposed_solution/RL_agent/evaluate.py`
17. Run LLM judge evaluation: `python proposed_solution/RL_agent/evaluate_llm.py`
18. Run demo: `python proposed_solution/RL_agent/demo.py`

---
