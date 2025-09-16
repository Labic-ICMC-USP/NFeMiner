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

---

## **How to Use**

⚙️ **Optional Requirement – Elasticsearch**  
If you want to use the indexing and search features, make sure you have **Elasticsearch** installed and running.  
By default, the library assumes the standard Elasticsearch configuration.  
If your instance does **not** use the default values, configure the following environment variables:

- `ELASTICSEARCH_HOST` (str): Elasticsearch server hostname or IP. Default: `localhost`.  
- `ELASTICSEARCH_PORT` (int): Elasticsearch server port. Default: `9200`.  
- `ELASTICSEARCH_SCHEME` (str): Protocol scheme (`http` or `https`). Default: `http`.  

---

## **📊 Example Dataset**

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

## **Example in Python**

Here is a minimal example of how to use **NFeMiner** in Python:

```python
import pandas as pd
from nfeminer import NFeMiner
from nfeminer.enrichment import NFeMinerLocalModel, NFeMinerGPTModel

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
# model = NFeMinerGPTModel(api_key="your-api-key")

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

print("\n🔎 Search results for numeric query (quantity_sold = 50.0):")
print(nfem.search_numeric_term('quantity_sold', 50.0))

# ===========================
# Step 4 - GTIN Estimation
# ===========================

# Training data: always use description + gtin_code from the DataFrame
training_data = df[['description', 'gtin_code']].values.tolist()

# Option A - Use descriptions from the DataFrame for classification
classify_data_df = df['description'].tolist()
results_df = nfem.gtin_estimator(
    training_description_gtin=training_data,
    classify_descriptions=classify_data_df
)

print("\n📦 GTIN Estimation (classify from DataFrame):")
print(pd.DataFrame(results_df))

# Option B - Use a custom list of descriptions for classification
custom_descriptions = [
    "Organic brown rice 1kg",
    "Sparkling water lemon flavor 500ml",
    "Semi-skimmed milk 1L"
]
results_custom = nfem.gtin_estimator(
    training_description_gtin=training_data,
    classify_descriptions=custom_descriptions
)

print("\n📦 GTIN Estimation (classify from custom list):")
print(pd.DataFrame(results_custom))

# ===========================
# Step 5 - Clustering
# ===========================

# Option A - Provide a custom list of product descriptions
custom_descriptions = [
    "Fresh beef hindquarter",
    "Beef shoulder",
    "Beef chops",
    "Long-grain rice",
    "Whole milk UHT 1L",
    "Sparkling water 500ml"
]

clusters_custom = nfem.clustering(custom_descriptions)
print("\n🧩 Clustering result (manual list):")
print(pd.DataFrame(clusters_custom))

# Option B - Use the 'description' column from the dataset (DataFrame)
df_descriptions = df['description'].tolist()
clusters_df = nfem.clustering(df_descriptions)

print("\n🧩 Clustering result (from DataFrame column):")
print(pd.DataFrame(clusters_df))
```