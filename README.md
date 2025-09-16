# **NFeMiner – Electronic Invoice Mining**

## **Processing Flow in NFeMiner**

NFeMiner follows a **modular cycle**, where each step receives pre-processed data and generates a structured output for subsequent modules.  
The general cycle can be divided into the following stages:

---

### **1️⃣ Enrichment and Disambiguation (Module: NFeEnrichment)**
📥 **Input:**  
- Raw NF-e documents in JSON format, with sensitive information anonymized.  

🛠 **Process:**  
- Uses **teacher LLMs** to **correct, standardize, and enrich** product descriptions.  
- Fills missing fields, harmonizes measurement units, and standardizes terms in descriptions.  

📤 **Output:**  
- Enriched NF-e documents in structured JSON format.  
- Refined semantic information, making it easier to compare similar products.  

---

### **2️⃣ Local Model Fine-Tuning (Module: LLMFineTuning)**
📥 **Input:**  
- Enriched data from the previous module.  

🛠 **Process:**  
- Performs **fine-tuning** of a compact **student LLM** using enriched data.  
- The student model learns to enrich new NF-e locally, reducing dependence on the teacher LLM.  

📤 **Output:**  
- Locally trained model for efficient processing of new invoices.  
- Reduced reliance on large, remote LLMs.  

---

### **3️⃣ Graph Construction and Clustering (Modules: NFeClustering + SimilarityGraph)**  
📥 **Input:**  
- Enriched and standardized NF-e data.  

🛠 **Process:**  
- Generates **edges between invoices** using multiple criteria:  
  - **StringMatchEdgeGenerator** → textual similarity.  
  - **BERTEmbeddingEdgeGenerator** → semantic similarity with Sentence-BERT.  
  - **NCMSimilarityEdgeGenerator** → shared NCM code.  
  - **ValueRangeEdgeGenerator / PriceBandEdgeGenerator** → proximity within numeric value ranges.  
- Builds a **NetworkX graph** with nodes = NF-e and edges = similarity relations.  
- Applies **community detection algorithms** (clique, label propagation, modularity, etc.)  
  to form product clusters.  

📤 **Output:**  
- **Clusters of “same product”**, labeled and ready for statistical analysis.  

---

### **4️⃣ GTIN Estimation (Module: NFeGtinEstimator)**
📥 **Input:**  
- Clustered and enriched invoices.  

🛠 **Process:**  
- Filters invoices with valid GTIN and high similarity to create reliable training sets.  
- Trains a classifier pipeline (string matching → TF-IDF/BoW + 1-NN → SBERT embeddings).  

📤 **Output:**  
- Assigns GTINs to missing entries with confidence scores and decision rules.  

---

### **5️⃣ Indexing in Elasticsearch (Module: ElasticSearchIndexer)**
📥 **Input:**  
- Enriched JSON, cluster labels, and GTIN estimates.  

🛠 **Process:**  
- Indexes data into **Elasticsearch** for fast and scalable search.  
- Structures documents for optimized queries across multiple attributes.  

📤 **Output:**  
- Data ready for exploration in Kibana and analysis via LLM agents.  

---

### **6️⃣ Exploration and Visualization (Kibana)**
📥 **Input:**  
- Indexed invoices in Elasticsearch.  

🛠 **Process:**  
- Interactive **dashboards in Kibana** to explore the data.  
- Visualization of product clusters and price trends.  

📤 **Output:**  
- Graphical interface (Kibana) for detailed analysis of products and average prices.  

---

## **Summary of Workflow and Main Modules**
1️⃣ **NFeEnrichment** → Enrichment and disambiguation of invoices.  
2️⃣ **LLMFineTuning** → Fine-tuning of a student LLM for local processing.  
3️⃣ **NFeClustering + SimilarityGraph** → Graph construction and clustering of similar invoices.  
4️⃣ **NFeGtinEstimator** → GTIN estimation using ML pipelines.  
5️⃣ **ElasticSearchIndexer** → Indexing enriched data into Elasticsearch.  
6️⃣ **ElasticSearchExplorer** → Visual exploration of invoices via Kibana.  
