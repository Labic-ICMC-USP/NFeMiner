from .enrichment import NFeMinerBaseGenerateModel
from .elasticsearch import NFeMinerElasticSearch
from .classification import NFeMinerGTINEstimator, NFeMinerModelCreator
from .similarity import SimilarityEngine, SequenceMatchSimilarity, NCMSimilarity, CategorySimilarity, TagSimilarity, BERTSimilarity
from .clustering import NFeMinerClustering
from .storage import KVStore
from typing import Union, Optional, List, Dict

class NFeMiner:
    """
    Facade for semantic enrichment of NFe (Brazilian electronic invoice) items using a pluggable language model.

    This class wraps an instance of a language model compatible with the `NFeMinerBaseGenerateModel` interface and
    extends its functionality with GTIN estimation, product clustering, and integration with Elasticsearch
    for indexing and searching enriched data.

    Attributes:
        model (NFeMinerBaseGenerateModel): The language model used for generating structured outputs.
        elasticsearch (NFeMinerElasticSearch): Interface for indexing and searching enriched documents.
        estimator (NFeMinerGTINEstimator): Classifier for predicting GTIN from item descriptions.
        clusterizer (NFeCluster): Component for clustering product descriptions.

    Args:
        model (NFeMinerBaseGenerateModel): Instance of a compatible model that implements `json_generate`.
        index_file_path (str): Path to a JSON or CSV file containing products to index. If None, indexing is skipped.
        index_name (str, optional): Name of the Elasticsearch index to use. Defaults to "nfe_products".
    """

    def __init__(self, model: NFeMinerBaseGenerateModel, index_file_path: str, index_name="nfe_products"):
        self.model = model
        if index_file_path is not None:
            self.elasticsearch = NFeMinerElasticSearch()
            self.index_name = index_name
            self.elasticsearch.index_service.create_index(self.index_name, index_file_path)
        else:
            self.elasticsearch = None

    def enrichment(self, invoice_id: str, item_id: str, ncm_code: str, gtin_code: str,
                   sales_unit: str, quantity_sold: float, unit_price: float, description: str) -> dict:
        """
        Performs semantic enrichment using the model with structured input fields.

        Args:
            invoice_id (str): Unique identifier of the electronic invoice (NFe).
            item_id (str): Identifier of the item within the invoice.
            ncm_code (str): Mercosur Common Nomenclature (NCM) code for tax classification.
            gtin_code (str): Global Trade Item Number (GTIN), e.g., barcode.
            sales_unit (str): Unit of measure used for selling the item (e.g., "kg", "pcs").
            quantity_sold (float): Quantity of the item sold.
            unit_price (float): Price per unit of the item.
            description (str): Natural language description of the product.

        Returns:
            dict: Structured enrichment output returned by the model.
        """
        return self.model.json_generate(invoice_id, item_id, ncm_code, gtin_code, sales_unit,
                                        quantity_sold, unit_price, description)

    def enrichment_and_index(self, raw_data: Union[Dict, List[Dict]], batch_size: int = 500) -> Dict[str, str]:
        """
        Performs enrichment on raw input data and indexes the result into Elasticsearch.

        Args:
            raw_data (Union[Dict, List[Dict]]): Single or list of dictionaries containing input fields.
            batch_size (int, optional): Number of documents to index per batch. Defaults to 500.

        Returns:
            Dict[str, str]: Dictionary with success message and document count.

        Raises:
            ValueError: If the input format is invalid or required fields are missing.
        """
        if isinstance(raw_data, dict):
            enriched_data = [self.enrichment(**raw_data)]
        elif isinstance(raw_data, list):
            enriched_data = [self.enrichment(**item) for item in raw_data]
        else:
            raise ValueError("Input data must be a dictionary or list of dictionaries.")

        return self.elasticsearch.document_service.index_documents(
            self.index_name, enriched_data, batch_size=batch_size
        )

    def search_string(self, value: str) -> dict:
        """
        Performs full-text search across all fields in the index.

        Args:
            value (str): Search string.

        Returns:
            dict: Search results from Elasticsearch.
        """
        query = {
            "query": {
                "query_string": {
                    "query": value
                }
            }
        }
        return self.elasticsearch.document_service.search_documents(self.index_name, query)

    def search_all(self) -> dict:
        """
        Retrieves all documents stored in the index.

        Returns:
            dict: All indexed documents.
        """
        query = {
            "query": {
                "match_all": {}
            }
        }
        return self.elasticsearch.document_service.search_documents(self.index_name, query)

    def search_numeric_term(self, field: str, value: Union[int, float]) -> dict:
        """
        Searches for documents with an exact numeric value in the specified field.

        Args:
            field (str): Field name to filter.
            value (int | float): Exact value to match.

        Returns:
            dict: Matching documents.
        """
        query = {
            "query": {
                "term": {
                    field: value
                }
            }
        }
        return self.elasticsearch.document_service.search_documents(self.index_name, query)

    def search_numeric_range(self, field: str, gte: Optional[float] = None, lte: Optional[float] = None) -> dict:
        """
        Filters documents where a numeric field falls within a given range.

        Args:
            field (str): Field name to filter.
            gte (float, optional): Minimum value (inclusive).
            lte (float, optional): Maximum value (inclusive).

        Returns:
            dict: Matching documents.
        """
        range_query = {}
        if gte is not None:
            range_query["gte"] = gte
        if lte is not None:
            range_query["lte"] = lte

        query = {
            "query": {
                "range": {
                    field: range_query
                }
            }
        }
        return self.elasticsearch.document_service.search_documents(self.index_name, query)

    def search_combined(self, must: List[dict] = [], should: List[dict] = [], must_not: List[dict] = []) -> dict:
        """
        Executes a boolean query with must, should, and must_not clauses.

        Args:
            must (List[dict]): Conditions that must be satisfied.
            should (List[dict]): Optional conditions that boost relevance.
            must_not (List[dict]): Conditions that must be excluded.

        Returns:
            dict: Search results from Elasticsearch.
        """
        query = {
            "query": {
                "bool": {
                    "must": must,
                    "should": should,
                    "must_not": must_not
                }
            }
        }
        return self.elasticsearch.document_service.search_documents(self.index_name, query)

    def gtin_estimator(self, training_description: list[str], training_gtin: list[str], classify_descriptions: list[str]) -> List[dict]:
        """
        Predict GTIN values for multiple product descriptions.

        Args:
            training_description (list[str]):
                List of training descriptions used to train the model.
                Example: training_description_gtin = ['Carne bovina','Carne porco']

            training_gtin (list[str]):
                List of training gtin used to train the model.
                Example: training_gtin = ['465465','4654564']

            classify_descriptions (list[str]):
                List of unlabeled product descriptions for classification.
                Example:
                    classify_descriptions = ['carne bovina alcatra', 'porco dianteiro']

        Returns:
            List[dict]: List of dictionaries containing GTIN classification results.
        """
        import pandas as pd

        # Dataframes
        training = pd.DataFrame({'original': training_description, 'gtin': training_gtin})
        unlabeled = pd.DataFrame({'original': classify_descriptions,'gtin': pd.NA})
        
        # Training Model
        try:
            NFeMinerModelCreator(data=training)
        except Exception as e:
            raise Exception(f"The model couldn't be create!!\n\n{str(e)}")

        # Classify unlabeled descriptions
        try:
            estimator = NFeMinerGTINEstimator(batch=unlabeled)
            return estimator.results['gtin'].tolist()
        except Exception as e:
            raise Exception(f"The model couldn't be create!!\n\n{str(e)}")

    def clustering(self, index: List, raw_descriptions: List[str], short_description: List[str]=None, full_description: List[str]=None, sales_unit: List[str]=None, ncm: List[str]=None, tags: List[List[str]]=None, category: List[List[str]]=None, tmp_files_path="./similarity_cache") -> dict:
        """
        Clusters product data based on semantic and attribute similarity using a
        graph-based community detection approach.

        This method builds a similarity graph from the provided product attributes
        using multiple similarity functions (e.g., BERT embeddings, sequence matching,
        and domain-specific comparators), applies clustering via `NFeMinerClustering`,
        and returns cluster assignments mapped to the original item IDs.

        Args:
            index (List):
                List of original IDs associated with each item. Must match the length
                of the input attribute lists. These IDs are used to map clustering
                results back to the original dataset.

            raw_descriptions (List[str]):
                Base list of product descriptions used as the primary textual feature.

            short_description (List[str], optional):
                Alternative short descriptions. When provided, they override
                `semantic_key` for semantic similarity.

            full_description (List[str], optional):
                Extended product descriptions. Used for additional similarity signals
                and may override `semantic_key` if present.

            sales_unit (List[str], optional):
                Sales unit information (e.g., "kg", "unit"). Used for lexical similarity.

            ncm (List[str], optional):
                NCM (Mercosur Common Nomenclature) codes. Used for domain-specific
                similarity comparisons.

            tags (List[List[str]], optional):
                List of tag sets associated with each item. Used for tag-based similarity.

            category (List[List[str]], optional):
                Hierarchical category information. Used for category-based similarity.

            tmp_files_path (str, optional):
                Directory used to store LMDB cache files. Defaults to
                `"./similarity_cache"`. This cache enables scalable pairwise similarity
                computation with constrained memory usage.

        Returns:
            dict[index, Dict[str, Any]]:
                Mapping from each item ID to its clustering metadata:
                {
                    "clustering": {
                        "lexical_group": "LEX-042",
                        "product_group": "PRD-169482",
                        "product_hierarchy": ["0001", "0001.0003"]
                    }
                }

                The fields `lexical_group` and `product_group` may be omitted when
                `lsh_key` is not defined.
        """

        funcs = [BERTSimilarity('raw_descriptions'), SequenceMatchSimilarity('raw_descriptions')]

        from pandas import DataFrame
        items = DataFrame(raw_descriptions, columns=["raw_descriptions"])

        semantic_key = "raw_descriptions"
        if full_description:
            funcs += [BERTSimilarity('full_description'), SequenceMatchSimilarity('full_description')]
            items["full_description"] = full_description
            semantic_key = "full_description"

        if short_description:
            funcs += [BERTSimilarity('short_description'), SequenceMatchSimilarity('short_description')]
            items["short_description"] = short_description
            semantic_key = "short_description"

        if sales_unit:
            funcs.append(SequenceMatchSimilarity('sales_unit'))
            items["sales_unit"] = sales_unit
        
        if ncm:
            funcs.append(NCMSimilarity('ncm'))
            items["ncm"] = ncm
        
        if category:
            funcs.append(CategorySimilarity('category'))
            items["category"] = category
        
        if tags:
            funcs.append(TagSimilarity('tags'))
            items["tags"] = tags

        items = items.to_dict(orient='records')

        map_size_bytes = 1 # up to 255 unique string keys (function names)
        cache = KVStore(path=tmp_files_path, key_mode=KVStore.KeyMode.SINGLE_KEY, value_mode=KVStore.ValueMode.NUMERIC, map_size_bytes=map_size_bytes)
        engine = SimilarityEngine(funcs=funcs, cache=cache, max_workers=None)

        rounds = {
            "algorithms":   ["louvain", "leiden"],
            "thresholds":   [0.6, 0.75, 0.9],
            "func_groups":  [engine.registered_functions()],
            "bootstrap":    None,
        }
        nfe = NFeMinerClustering(items=items, ids=index, engine=engine, rounds=rounds, n_runs_per_round=3, max_depth=3, min_cluster_size=2, lsh_key="raw_descriptions", semantic_key=semantic_key)

        return nfe.run()