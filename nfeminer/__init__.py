from .enrichment import NFeMinerBaseGenerateModel, NFeMinerJSONValidator
from .elasticsearch import NFeMinerElasticSearch
from .classification import NFeMinerGTINEstimator, NFeModelCreator
from .clustering import NFeCluster
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

    def gtin_estimator(self, training_description_gtin: list[str], classify_descriptions: list[list[str, str]]) -> List[dict]:
        """
        Predict GTIN values for multiple product descriptions.

        Args:
            training_description_gtin (list[list[str]]):
                List of training pairs [description, gtin] used to train the model.
                Example:
                    training_description_gtin = [
                        ['Carne bovina', '465465'],
                        ['Carne porco', '4654564']
                    ]

            classify_descriptions (list[str]):
                List of unlabeled product descriptions for classification.
                Example:
                    classify_descriptions = ['carne bovina alcatra', 'porco dianteiro']

        Returns:
            List[dict]: List of dictionaries containing GTIN classification results.

        Notes:
            - Adjust GPU parameters, thread count, and classification threshold as needed.
        """
        import pandas as pd

        # Dataframes
        training = pd.DataFrame(training_description_gtin, columns=['original', 'gtin'])
        unlabeled = pd.DataFrame({'original': classify_descriptions,'gtin': pd.NA})
        
        # Training Model
        try:
            NFeModelCreator(data=training)
        except Exception as e:
            print('Erro na criação do modelo')
            return []

        # Classify unlabeled descriptions
        try:
            estimator = NFeMinerGTINEstimator(batch=unlabeled)
        except Exception as e:
            print('Erro na classificação')
            return []

        # Return results
        return estimator.results['gtin'].tolist()

    def clustering(self, descriptions: List[str]) -> dict:
        """
        Clusters product descriptions using semantic similarity.

        Args:
            descriptions (List[str]): List of product descriptions.

        Returns:
            label2descs (dict[int,list[str]]): Dictionary mapping cluster labels to grouped descriptions.
        """
        nfc = NFeCluster(data=descriptions)
        only_labels, clusterized = nfc.cluster()
        
        label2descs = {}
        for desc,label in list(zip(descriptions,only_labels)):
            label = int(label)
            if label in label2descs.keys():
                label2descs[label].append(desc)
            else:
                label2descs[label] = [desc,]
        return label2descs
