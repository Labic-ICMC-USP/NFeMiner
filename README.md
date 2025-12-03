# **NFeMiner 2.0 – Electronic Invoice Mining**

**NFeMiner** is a framework for processing and analyzing Brazilian Electronic Invoices (NF-e).
It integrates **semantic enrichment with Large Language Models (LLMs)**, **graph-based clustering**, **GTIN estimation**, and **efficient indexing** to produce enriched, structured data ready for analytics and search.

## **The Problem**

NF-e documents often contain **missing fields, inconsistent product descriptions, ambiguous terminology, and a lack of standardization**.
These issues make it difficult to:

* Identify **invoices referring to the same product**,
* Perform reliable **statistical analysis**,
* Compute aggregated metrics such as **average price**,
* And integrate data across heterogeneous sources.

### **How NFeMiner Solves It**

NFeMiner addresses these challenges by applying:

* **Semantic enrichment** (LLMs or local fine-tuned models) to correct, normalize, and complete NF-e product descriptions.
* **Similarity-based clustering** to group invoices that refer to the *same underlying product*.
* **GTIN estimation** to infer missing identifiers using hybrid matching and machine-learning pipelines.
* **Indexing and search capabilities** through Elasticsearch for scalable exploration and integration with tools like Kibana.

The result is a clean, enriched, standardized dataset that supports robust analytics, interoperability, and automated downstream processing.

---

## **NFeMiner Architecture and Processing Flows**

NFeMiner is organized into **five independent modules**, each handling a specific aspect of NF-e normalization, enrichment, clustering, GTIN estimation, and indexing.
These modules can be used together or independently, depending on the workflow.
Although the system supports multiple strategies, the processing ultimately follows **two major flows**:

1. **Teacher–Student Enrichment Flow** (LLM-based enrichment → local fine-tuning)
2. **Operational Processing Flow** (local or remote LLMs → enrichment → clustering → GTIN estimation → indexing)

---

### **A. Teacher–Student Enrichment Flow**

This flow is used when the goal is to **extract high-quality enriched data using powerful “teacher” LLMs**, and then **train compact local “student” models** capable of performing the same enrichment offline.

1. NF-e raw documents are enriched using one of the **Teacher Enrichment Engines**
   (Ollama, OpenRouter, or Local-LLMs).
2. The enriched dataset becomes a high-quality corpus.
3. The **Fine-Tuning Module** trains a smaller local model that replicates the teacher’s output.
   This model can later be deployed for offline, cost-efficient enrichment.

This flow is ideal for building high-precision, self-contained enrichment pipelines.

---

### **B. Operational Processing Flow**

This is the end-to-end flow followed in production environments:

1. NF-e documents (raw or already normalized) are enriched using any available LLM strategy
   (Ollama, OpenRouter, or local fine-tuned models).
2. The enriched documents are fed to the **Clustering Module**, which groups invoices that refer to the same underlying product.
3. The **GTIN Estimation Module** receives invoices with valid and missing GTINs and assigns a GTIN prediction to incomplete entries.
4. The final dataset (containing enriched descriptions, cluster IDs, and GTIN estimates) is sent to the **Elasticsearch Integration Module** for fast search, analytics, and downstream applications.

This flow is modular and can be executed with local or remote LLMs, depending on the environment and available compute.

---

### **Enrichment Module**

**Classes:**

* `NFeMinerOpenRouterModel`
* `NFeMinerOllamaModel`
* `NFeMinerLocalModel`

**Input:**
Raw NF-e documents (JSON), with sensitive fields anonymized.

**Process:**

* Uses an LLM (teacher or local model) to normalize, correct, and enrich product descriptions.
* Standardizes terminology, harmonizes measurement units, and fills missing fields.

**Output:**
Structured, enriched NF-e documents with refined semantic and numeric attributes.

---

### **Fine-Tuning Module**

**Class:**

* `NFeFinetuner`

**Input:**

* Enriched NF-e dataset
* Base model to be fine-tuned

**Process:**

* Trains a compact local “student” model to replicate the enrichment behavior of a stronger teacher LLM.
* Produces a reproducible fine-tuned model for efficient offline inference.

**Output:**
A fine-tuned local model capable of enriching new invoices without external LLM calls.

---

### **Clustering Module**

**Classes:**

* `NFeCluster`
* `StringMatchEdgeGenerator`
* `BERTEmbeddingEdgeGenerator`
* `NCMSimilarityEdgeGenerator`
* `PriceBandEdgeGenerator`
* `ValueRangeEdgeGenerator`

**Input:**

* Enriched (or optionally raw) NF-e descriptions

**Process:**

* Computes similarity edges using a defined strategy
* Builds a similarity graph and applies community-detection algorithms (e.g., Louvain, LPA).
* Assigns each invoice to a product cluster.

**Output:**
Cluster labels representing groups of invoices referring to the same underlying product.

---

### **GTIN Estimation Module**

**Classes:**

* `NFeMinerModelCreator`
* `NFeMinerGTINEstimator`

**Input:**

* Invoices **with GTIN**
* Invoices **without GTIN**
* Inputs may be enriched or raw

**Process:**

* Selects high-quality training pairs from invoices with valid GTINs.
* Uses a hybrid pipeline (string matching → TF-IDF/BOW + 1-NN → SBERT embeddings).
* Predicts the most likely GTIN for invoices with missing codes.

**Output:**
GTIN predictions with confidence scores, integrated into the enriched NF-e documents.

---

### **Elasticsearch Integration Module**

**Class:**

* `NFeMinerElasticSearch`

**Input:**

* Final enriched dataset with cluster labels and GTIN predictions

**Process:**

* Creates and manages Elasticsearch indices.
* Indexes new invoices, updates existing entries, and supports deletions.
* Offers optimized search operations and analytical queries.

**Output:**
Search-ready NF-e database accessible for dashboards, analytics, and LLM-based agents.

---

## **Installation**

To install the NFeMiner library, we recommend using **uv**, a fast and modern Python package manager that provides deterministic environments and manages dependencies, ensuring compatibility between NFeMiner and your existing environment.

A typical installation flow with uv looks like this:

```bash
# Initialize a new project 
# (if you already have a project, skip this step)
uv init my-nfeminer-project
cd my-nfeminer-project

# Add NFeMiner as a dependency (GPU version)
uv add "./NFeMiner[gpu]"

# Alternatively, install the CPU-only version.
# Use this if:
# - you do not need GPU acceleration, OR
# - the GPU dependencies are already installed, OR
# - the GPU dependencies caused conflicts in your environment
uv add "./NFeMiner"

# Install and synchronize all dependencies
uv sync
```

### **Alternative Installation Using pip**

You can also install NFeMiner using `pip`.

```bash
# Install NFeMiner from a local directory (CPU version)
pip install ./NFeMiner

# Install NFeMiner from a local directory with GPU dependencies
pip install "./NFeMiner[gpu]"
```

---

## **How to Use**

The repository [NFeMiner-Docker](https://github.com/Labic-ICMC-USP/NFeMiner-docker/) provides a preconfigured Docker environment that uses the NFeMiner library. This is the easiest and most complete way to run NFeMiner, as it already includes Elasticsearch, Kibana, and a web interface that allows you to upload data, perform LLM-based enrichment, and index the results directly into Elasticsearch.

In addition, the script folder contains more complete example codes demonstrating how to use the NFeMiner modules for [Enrichment](https://github.com/Labic-ICMC-USP/NFeMiner-docker/tree/main/scripts), [Finetuning](https://github.com/Labic-ICMC-USP/NFeMiner-docker/tree/main/scripts), [Classification](https://github.com/Labic-ICMC-USP/NFeMiner-docker/tree/main/scripts), and [Clustering](https://github.com/Labic-ICMC-USP/NFeMiner-docker/tree/main/scripts). These examples can be found in the repository [NFeMiner-Docker in the scripts folder](https://github.com/Labic-ICMC-USP/NFeMiner-docker/tree/main/scripts), specifically in the files prefixed with **job_**.

Below we provide a simple usage example intended only to illustrate the basic use of the functions.

⚙️ **Optional Requirement – Elasticsearch**
If you want to use the indexing and search features, make sure you have **Elasticsearch** installed and running.
By default, the library assumes the standard Elasticsearch configuration.
If your instance does **not** use the default values, configure the following environment variables:

- `ELASTICSEARCH_HOST` (str): Elasticsearch server hostname or IP. Default: `localhost`.
- `ELASTICSEARCH_PORT` (int): Elasticsearch server port. Default: `9200`.
- `ELASTICSEARCH_SCHEME` (str): Protocol scheme (`http` or `https`). Default: `http`.

---

### **📊 Example Dataset**

Use the following data structure to represent invoices.

| invoice_id | item_id | ncm_code | gtin_code  | sales_unit | quantity_sold | unit_price | description                 |
|------------|---------|----------|------------|------------|---------------|------------|-----------------------------|
| INV001     | 1       | 10061010 | 7891000051 | kg         | 50.0          | 4.20       | Long-grain white rice 5kg   |
| INV002     | 1       | 22021000 | 7894900012 | pcs        | 200.0         | 2.50       | Sparkling water 500ml       |
| INV002     | 2       | 04031000 | 7897700025 | l          | 100.0         | 6.90       | Whole milk UHT 1L           |

**Column Descriptions:**
- `invoice_id (str)` → Unique identifier of the electronic invoice (NFe).
- `item_id (str)` → Identifier of the item within the invoice.
- `ncm_code (str)` → Mercosur Common Nomenclature (NCM) code for tax classification.
- `gtin_code (str)` → Global Trade Item Number (GTIN), e.g., barcode.
- `sales_unit (str)` → Unit of measure used for selling the item (e.g., "kg", "pcs").
- `quantity_sold (float)` → Quantity of the item sold.
- `unit_price (float)` → Price per unit of the item.
- `description (str)` → Natural language description of the product.

---

### **Example in Python**

Here is a minimal example of how to use **NFeMiner** in Python:

```python
import pandas as pd
from nfeminer import NFeMiner
from nfeminer.enrichment import NFeMinerLocalModel, NFeMinerOpenRouterModel

# ===========================
# Example dataset (3 invoices)
# ===========================
invoices = [
    {
        "invoice_id": "INV001",
        "item_id": "1",
        "ncm_code": "10061010",
        "gtin_code": "7891000051",
        "sales_unit": "kg",
        "quantity_sold": 50.0,
        "unit_price": 4.20,
        "description": "Long-grain white rice 5kg"
    },
    {
        "invoice_id": "INV002",
        "item_id": "1",
        "ncm_code": "22021000",
        "gtin_code": "7894900012",
        "sales_unit": "pcs",
        "quantity_sold": 200.0,
        "unit_price": 2.50,
        "description": "Sparkling water 500ml"
    },
    {
        "invoice_id": "INV002",
        "item_id": "2",
        "ncm_code": "04031000",
        "gtin_code": "7897700025",
        "sales_unit": "l",
        "quantity_sold": 100.0,
        "unit_price": 6.90,
        "description": "Whole milk UHT 1L"
    }
]

df = pd.DataFrame(invoices)

# ===========================
# Choose the model
# ===========================
# Option 1 - Local fine-tuned model
model = NFeMinerLocalModel()

# Option 2 - GPT-based teacher model
# model = NFeMinerOpenRouterModel(api_key="your-api-key")

# ===========================
# Initialize NFeMiner
# ===========================
nfem = NFeMiner(model, './index.mapping.json')

# ===========================
# Step 1 - Enrichment (single row)
# ===========================
row = df.iloc[0]
enriched = nfem.enrichment(
    invoice_id=row['invoice_id'],
    item_id=row['item_id'],
    ncm_code=row['ncm_code'],
    gtin_code=row['gtin_code'],
    sales_unit=row['sales_unit'],
    quantity_sold=row['quantity_sold'],
    unit_price=row['unit_price'],
    description=row['description']
)
print("🔹 Enrichment result (single row):")
print(enriched)

# ===========================
# Step 2 - Enrichment + Indexing (whole dataset)
# ===========================
nfem.enrichment_and_index(df.to_dict(orient='records'))
print("✅ Dataset enriched and indexed into Elasticsearch (if configured).")

# ===========================
# Step 3 - Search
# ===========================
print("\n🔎 Search results for 'rice':")
print(nfem.search_string('rice'))

print("\n🔎 Search results for numeric query (quantidade_comercializada = 50.0):")
print(nfem.search_numeric_term('quantidade_comercializada', 50.0))

# ===========================
# Step 4 - GTIN Estimation
# ===========================

# Training data: always use description + gtin_code from the DataFrame
training_data = df[['description', 'gtin_code']]

# Option A - Use descriptions from the DataFrame for classification
# unlabeled_data = df['description'].tolist()

# Option B - Use a custom list of descriptions for classification
unlabeled_data = [
    "Organic brown rice 1kg",
    "Sparkling water lemon flavor 500ml",
    "Semi-skimmed milk 1L"
]

results_df = nfem.gtin_estimator(
    training_description=training_data['description'].tolist(),
    training_gtin=training_data['gtin_code'].tolist(),
    classify_descriptions=unlabeled_data
)

print("\n📦 GTIN Estimation (classify):")
print(pd.DataFrame(results_df))

# ===========================
# Step 5 - Clustering
# ===========================

# Option A - Use descriptions from the DataFrame
descriptions = df['description'].tolist()
index = df.index.tolist()

# Option B - Provide a custom list of product descriptions based on the enriched and indexed data
# results_search = nfem.search_string('rice')
# hits = results_search.get("hits", [])
# index = []
# descriptions = []
# for doc in hits:
#     index.append(doc["_id"])
#     enriched = doc.get("descricao", {}).get("enriquecida", {}).get("produto_detalhado")
#     descriptions.append(enriched)

clusters_custom = nfem.clustering(descriptions, index)
print("\n🧩 Clustering result (manual list):")
print(pd.DataFrame(clusters_custom))
```