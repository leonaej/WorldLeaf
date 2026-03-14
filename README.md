# WorldLeaf

Hand-curated African Savanna knowledge graph with 1,423 species nodes and 1,541 ecological edges, built to support an interpretable reinforcement learning agent for multi-hop ecological reasoning. Master's thesis project.

---

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
├── Baseline/
│   ├── fetch_wikipedia.py       # Wikipedia article fetching via Wikidata QIDs
│   ├── embed_nodes.py           # Wikipedia embeddings (text-embedding-3-large)
│   ├── embed_edges.py           # Graph relationship embeddings
│   └── save_edge_texts.py       # Save edge texts to JSON
├── RAG/
│   ├── retriever.py             # Cosine similarity retrieval over 6144d index
│   └── rag_pipeline.py          # End-to-end RAG answer generation
├── QA/
│   ├── generate_single_hop.py   # Single-hop QA pair generation from edges
│   ├── build_chains.py          # Multi-hop chain enumeration
│   └── generate_multi_hop.py    # Natural language question generation via GPT-4o mini
├── Evaluation/
│   └── evaluate_rag.py          # LLM-judged Hit@3 evaluation with failure mode breakdown
├── data/
│   ├── raw/                     # Raw edge and species files, not commited to git
│   └── processed/               # Final graph files and visualization
├── notebooks/
│   └── 01_explore_graph.ipynb   # Graph exploration notebook
├── misc/                        # Ad-hoc scripting folder (taxonomy enrichment)
├── visualize_graph.py           # Graph visualization
└── requirements.txt
```

---

## RAG Baseline

Each node is represented by two 3,072-dimensional embeddings, one from its Wikipedia text and one from a structured text of its graph relationships, both produced using OpenAI text-embedding-3-large. At retrieval time these are concatenated into a 6,144-dimensional vector. A query is embedded, compared via cosine similarity against all 1,423 nodes, and the top-3 nodes are retrieved. Their full text is passed as context to GPT-4o mini, which answers using only the provided context.

Keeping the two embeddings separate enables ablation studies at Stage 3.

### Evaluation Results (Hit@3, LLM-judged)

| Split | Hit | Type 1 (retrieval miss) | Type 2 (data gap) | Type 3 (wrong info) |
|---|---|---|---|---|
| Single-hop (2,276) | 89.6% | 4.0% | 6.3% | 0.0% |
| Multi-hop (8,405) | 38.2% | 12.2% | 49.5% | 0.1% |
| Overall (10,681) | 49.1% | 10.5% | 40.3% | 0.1% |

The strong single-hop performance but sharp multi-hop degradation confirms the core motivation: flat retrieval lacks the relational chaining needed for multi-hop ecological reasoning. The near-zero Type 3 rate validates knowledge graph data quality.

---

## QA Dataset
- The dataset is available <a href="https://drive.google.com/drive/folders/17X2n96Rr8LDyVdhGqN1Vz80HRhRMbsL7?usp=sharing" target="_blank">here</a>.
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
11. Run RAG evaluation: `python Evaluation/evaluate_rag.py`

---
