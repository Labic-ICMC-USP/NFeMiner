Introduction
============

Motivation
----------

Although **Electronic Invoices (NF-e)** are a rich source of data about commercial transactions, they often suffer from **information quality issues**. The most common problems include:

- **Missing data**  
  Critical fields such as **GTIN** (product barcode), **NCM** (fiscal classification), or **unit of measure** are often left blank.

- **Non-standardized descriptions**  
  The same product may appear in many different formats, for example:
  
  - ``RICE 5KG TYPE 1``  
  - ``RICE TYPE I PACK 5 K``  
  - ``RICE W/ 5 KG``  

- **Semantic inconsistencies**  
  Relevant information such as packaging, brand, or quantity may be distributed across different fields or ambiguously described.

These limitations have **direct consequences** for large-scale analysis:

- Difficulty in automatically identifying when **two NF-e records refer to the same product**, since descriptions vary by supplier or even due to typos.  
- Inaccurate calculation of **aggregate metrics**, such as the average price of an item, because the same product may be scattered across different records.  
- Lower quality of **Machine Learning models**, which rely on consistent attributes to learn reliable patterns.  
- Loss of trustworthiness in contexts such as **tax auditing**, **market research**, and **competitiveness analysis**.  

Objective
---------

The **NFeMiner** project aims to **restore the structure and identity of products** within NF-e records by means of:

- **Semantic enrichment** of descriptions (extracting packaging, measurements, brand, categories).  
- **Clustering of similar invoices** into graphs (clusters of the “same product”).  
- **GTIN estimation** when the code is missing or inconsistent.  
- **Efficient indexing and search** for analytical exploration (Elasticsearch/Kibana).  

Expected results
----------------

- **Deduplication/normalization** of equivalent items.  
- **Enriched JSON** validated for each item (Pydantic schema).  
- **Clusters** with labels assigned by community detection and label propagation algorithms.  
- **GTIN assignment** with confidence rules and multiple matching strategies.  
- **Fast querying** and **dashboards** for price and trend analysis.  

Main components
---------------

- **Classification (GTIN)** — ``NFeModelCreator`` and ``NFeMinerGTINEstimator``  
  - *String Matching*: dictionary of unique descriptions.  
  - *Vectorizers + 1-NN*: TF-IDF / BoW (word/char/char-n-gram).  
  - *Sentence-BERT*: semantic embeddings for similarity.  

- **Graphs and Clustering** — ``CGraph``, ``Analyser`` and *edge generators*  
  - Edge generation via **string match**, **BERT embeddings**, **NCM**, **value ranges**.  
  - NetworkX algorithms: *clique*, *label propagation*, *greedy modularity*, *bisection*, *girvan–newman*.  
  - **Semi-supervised** with label propagation and **group merging** for refinement.  

- **Enrichment** — ``NFeMinerBaseGenerateModel`` (+ local/OpenRouter/Ollama implementations)  
  - Generation of structured JSON (packaging, physical features, brand, origin, categories, tags).  
  - Validation schema with **Pydantic**.  

- **Elasticsearch** — ``NFeMinerElasticSearch``  
  - Facade for indexing, updating, deleting, and searching.  
  - Separate services for **documents** and **indices**.  

- **Fine-tuning** — ``NFeFinetuner``  
  - Pipeline with **PEFT** and **SFT** to train a local “student” model from enriched data.  

Edge generators (similarity_graph)
----------------------------------

- ``StringMatchEdgeGenerator`` — pairs with high similarity (``difflib.SequenceMatcher``).  
- ``BERTEmbeddingEdgeGenerator`` — pairs by semantic similarity (Sentence-BERT, cosine).  
- ``NCMSimilarityEdgeGenerator`` — pairs that share the same NCM.  
- ``ValueRangeEdgeGenerator`` / ``PriceBandEdgeGenerator`` — pairs within numeric bands/ranges.  
- ``build_graph`` — builds NetworkX graphs from node/edge DataFrames.  
