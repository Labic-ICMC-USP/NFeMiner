Processing Flow
===============

Overview
--------

NFeMiner follows a **modular cycle** where each stage produces artifacts that are consumed by the next ones:

1) **Enrichment and Disambiguation** (``enrichment``)  
   - Input: raw NF-e documents (JSON).  
   - Process: LLMs correct, standardize, and enrich product descriptions.  
   - Output: structured and validated JSON (Pydantic).  

2) **Local Model Fine-tuning** (``finetuning``)  
   - Input: enriched data.  
   - Process: SFT/PEFT to train a local “student” model.  
   - Output: local model capable of enriching future NF-e without depending on the “teacher” model.  

3) **Graph-based Clustering** (``clustering`` + ``similarity_graph``)  
   - Input: enriched/standardized invoices.  
   - Process: edge generation (string match, BERT, NCM, value ranges) and community detection.  
   - Output: clusters of “same product” with assigned labels.  

4) **GTIN Estimation** (``classification``)  
   - Input: clusters and reliable references.  
   - Process: *string match* → TF-IDF/BoW + 1-NN → SBERT embeddings (confidence pipeline).  
   - Output: estimated GTIN with similarity and decision method.  

5) **Indexing** (``elasticsearch``)  
   - Input: enriched JSON, cluster labels, and estimated GTIN.  
   - Process: index creation/update and document storage.  
   - Output: queryable data with low latency.  

6) **Exploration and Visualization** (Kibana)  
   - Input: Elasticsearch indices.  
   - Process: interactive dashboards (product groups, price series, etc.).  
   - Output: visual insights and analyses.  

.. important::
   Step 6 relies on **Kibana**, which is an **external service** (not a Python module within this package).
