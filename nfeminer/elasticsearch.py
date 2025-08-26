import json, os
from typing import List, Dict, Union
from elasticsearch import Elasticsearch, exceptions, helpers

class NFeMinerElasticSearch:
    """
    Facade for managing Elasticsearch operations related to NFe (Nota Fiscal Eletrônica) processing.
    
    This class provides a high-level interface to Elasticsearch for Brazilian electronic invoice (NFe)
    use cases. It abstracts both document-level and index-level operations, making it easier to manage
    indexing, updating, deletion, and search of structured fiscal data.

    Parameters:
        host (str): Hostname or IP address of the Elasticsearch server. Default is "localhost".
        port (int | str): Port number for Elasticsearch. Default is 9200.
        scheme (str): Protocol scheme ("http" or "https"). Default is "http".

    Attributes:
        client (Elasticsearch): The raw Elasticsearch client instance.
        document_service (ElasticSearchDocumentService): Document-level operations manager.
        index_service (ElasticSearchIndexService): Index-level operations manager.

    Methods:
        get_client() -> Elasticsearch:
            Returns the underlying Elasticsearch client.
    """

    def __init__(self):
        """
        Initializes the Elasticsearch client using environment variables.

        Environment Variables:
            ELASTICSEARCH_HOST (str): Elasticsearch server hostname or IP. Default: 'localhost'.
            ELASTICSEARCH_PORT (int): Elasticsearch server port. Default: 9200.
            ELASTICSEARCH_SCHEME (str): Protocol scheme ('http' or 'https'). Default: 'http'.

        Raises:
            RuntimeError: If the connection to Elasticsearch fails or configuration is invalid.
        """
        try:
            host = os.getenv("ELASTICSEARCH_HOST", "localhost")
            port = int(os.getenv("ELASTICSEARCH_PORT", 9200))
            scheme = os.getenv("ELASTICSEARCH_SCHEME", "http")

            self.client = Elasticsearch(hosts=[{"host": host, "port": port, "scheme": scheme}])

            if not self.client.ping():
                raise exceptions.ConnectionError("Elasticsearch is not reachable.")

            self.document_service = ElasticSearchDocumentService(self.client)
            self.index_service = ElasticSearchIndexService(self.client)

        except ValueError:
            raise RuntimeError(f"Invalid port: {os.getenv('ELASTICSEARCH_PORT')}. Port must be an integer.")
        except exceptions.ConnectionError as e:
            raise RuntimeError(f"Failed to connect to Elasticsearch: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while initializing Elasticsearch: {str(e)}")

    def get_client(self) -> Elasticsearch:
        """
        Returns the Elasticsearch client.

        Returns:
            Elasticsearch: The Elasticsearch client instance.
        """
        return self.client

class ElasticSearchDocumentService:
    """
    Manages document operations in Elasticsearch, including indexing, updating, deleting, and searching.
    """

    def __init__(self, client):
        """
        Initializes the Elasticsearch index manager with an existing Elasticsearch client.

        Args:
            connection (Elasticsearch): An instance of Elasticsearch class.
        """
        
        self.client = client

    def index_documents(self, index_name: str, documents: Union[Dict, List[Dict]], batch_size: int = 500, progress_callback=None) -> Dict[str, str]:
        """
        Indexes documents into Elasticsearch in batches, ignoring documents that raise errors.

        Args:
            index_name (str): Name of the Elasticsearch index.
            documents (Union[Dict, List[Dict]]): A single document (dict) or a list of documents (list[dict]).
            batch_size (int): Number of documents to send per batch.

        Returns:
            Dict[str, str]: A message indicating the number of indexed documents.

        Raises:
            ValueError: If the input data is not in the expected format.
            RuntimeError: If an error occurs during indexing (other than individual document errors).
        """
        try:
            if isinstance(documents, dict):
                documents = [documents]
            elif not isinstance(documents, list) or not all(isinstance(doc, dict) for doc in documents):
                raise ValueError("Invalid format: Expected a dictionary or a list of dictionaries.")

            total_indexed = 0
            failed_documents = []  # To track failed documents
            total_documents = len(documents)

            # Split documents for batch processing
            for i in range(0, total_documents, batch_size):
                batch = documents[i:i + batch_size]
                requests = [
                    {**doc, "_op_type": "index", "_index": index_name} for doc in batch
                ]
                # Use the helpers.bulk method to index the batch and ignore errors on individual documents
                success, failed = helpers.bulk(self.client, requests, raise_on_error=False)
                
                total_indexed += success
                failed_documents.extend(failed)
                if progress_callback:
                    progress_callback(total_indexed/total_documents)

            self.client.indices.refresh(index=index_name)
            return {
                "message": f"{total_indexed} documents were successfully indexed, while {len(failed_documents)} documents failed to be indexed.",
                "failed_documents": failed_documents
            }

        except Exception as e:
            raise RuntimeError(f"Unexpected error during index: {str(e)}")

    def update_document(self, index_name: str, doc_id: str, update_fields: Dict) -> Dict[str, str]:
        """
        Updates an existing document in Elasticsearch.

        Args:
            index_name (str): The name of the index.
            doc_id (str): The document ID to update.
            update_fields (Dict): The fields to update in the document.

        Returns:
            Dict[str, str]: A dictionary containing the update result message.

        Raises:
            RuntimeError: If an error occurs during the update operation.
        """
        try:
            response = self.client.update(index=index_name, id=doc_id, body={"doc": update_fields})
            return {"message": f"Document '{doc_id}' updated successfully.", "result": response["result"]}
        except Exception as e:
            raise RuntimeError(f"Unexpected error during update: {str(e)}")

    def delete_document(self, index_name: str, doc_id: str) -> Dict[str, str]:
        """
        Deletes a document from Elasticsearch.

        Args:
            index_name (str): The name of the index.
            doc_id (str): The document ID to delete.

        Returns:
            Dict[str, str]: A dictionary containing the delete result message.

        Raises:
            RuntimeError: If an error occurs during the delete operation.
        """
        try:
            response = self.client.delete(index=index_name, id=doc_id)
            return {"message": f"Document '{doc_id}' deleted successfully.", "result": response["result"]}
        except Exception as e:
            raise RuntimeError(f"Unexpected error during delete: {str(e)}")

    def search_documents(self, index_name: str, query: Dict) -> Dict[str, Union[List[Dict], str]]:
        """
        Searches for documents in Elasticsearch based on a query.

        Args:
            index_name (str): The name of the index.
            query (Dict): A dictionary containing the search query.

        Returns:
            Dict[str, Union[List[Dict], str]]: A dictionary with the search results. The key 'hits' contains a list of documents, or an error message if no results were found.

        Raises:
            RuntimeError: If an error occurs during the search operation.
        """
        try:
            response = self.client.search(index=index_name, body=query)
            hits = response['hits']['hits']
            if hits:
                return {
                    "hits": [{
                        **hit["_source"],
                        "_id": hit["_id"],
                        "_score": hit.get("_score"),
                        "_index": hit.get("_index"),
                    } for hit in hits]
                }
            else:
                return {"hits": [], "message": "No documents found matching the query."}
        except Exception as e:
            raise RuntimeError(f"Unexpected error during search: {str(e)}")

    def count_documents(self, index_name: str) -> int:
        """
        Count the documents in Elasticsearch.

        Args:
            index_name (str): The name of the index.

        Returns:
            int: 

        Raises:
            RuntimeError: If an error occurs during the search operation.
        """
        try:
            response = self.client.count(index=index_name, body={"query": {"match_all": {}}})
            print(response)
            return response["count"]
        except Exception as e:
            raise RuntimeError(f"Unexpected error during search: {str(e)}")

class ElasticSearchIndexService:
    """
    Manages Elasticsearch index operations, including creation, updating, and deletion.
    """

    def __init__(self, client: Elasticsearch):
        """
        Initializes the Elasticsearch index manager with an existing Elasticsearch client.

        Args:
            client (Elasticsearch): An instance of Elasticsearch class.
        """
        
        self.client = client

    def check_index_exists(self, index_name: str) -> Dict[str, str]:
        """
        Check if an Elasticsearch index exists and return a message.

        This function checks if the specified index exists in Elasticsearch and returns 
        a message indicating whether the index is present or not.

        Args:
            index_name (str): The name of the index to check.

        Returns:
            Dict[str, str]: A dictionary containing a message that indicates if the index 
                            exists or not.

        Raises:
            RuntimeError: If an error occurs while checking the index's existence.
        """
        try:
            if self.client.indices.exists(index=index_name):
                return {"message": f"Index '{index_name}' exists."}
            else:
                return {"message": f"Index '{index_name}' does not exist."}
        except Exception as e:
            raise RuntimeError(f"Error checking index existence: {str(e)}")
    
    def create_index(self, index_name: str, mapping: Union[str, Dict]) -> Dict[str, str]:
        """
        Creates an index in Elasticsearch with a provided mapping.

        Args:
            index_name (str): The name of the index to create.
            mapping (Union[str, Dict]): The mapping definition as a dictionary or a JSON string.

        Returns:
            Dict[str, str]: A dictionary containing the index creation result message.

        Raises:
            ValueError: If mapping is invalid.
            RuntimeError: If an error occurs during index creation.
        """
        try:
            if self.client.indices.exists(index=index_name):
                return {"error": f"Index '{index_name}' already exists."}

            if isinstance(mapping, str):
                try:
                    with open(mapping, "r") as f:
                        mapping = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON mapping: {str(e)}")

            if not isinstance(mapping, dict):
                raise ValueError("Mapping must be a dictionary or a valid JSON string.")

            self.client.indices.create(index=index_name, body=mapping)
            return {"message": f"Index '{index_name}' created successfully."}

        except Exception as e:
            raise RuntimeError(f"Error during index creation: {str(e)}")


    def update_index(self, index_name: str, settings: Dict) -> Dict[str, str]:
        """
        Updates an existing Elasticsearch index with new settings.

        Args:
            index_name (str): The name of the index to update.
            settings (Dict): The new settings to apply to the index.

        Returns:
            Dict[str, str]: A dictionary containing the index update result message.

        Raises:
            RuntimeError: If an error occurs during index update.
        """
        try:
            self.client.indices.put_settings(index=index_name, body=settings)
            return {"message": f"Index '{index_name}' updated successfully."}
        except Exception as e:
            raise RuntimeError(f"Unexpected error during index update: {str(e)}")

    def delete_index(self, index_name: str) -> Dict[str, str]:
        """
        Deletes an entire index from Elasticsearch.

        Args:
            index_name (str): The name of the index to delete.

        Returns:
            Dict[str, str]: A dictionary containing the index deletion result message.

        Raises:
            RuntimeError: If an error occurs during the index deletion operation.
        """
        try:
            if not self.client.indices.exists(index=index_name):
                return {"message": f"Index '{index_name}' does not exist."}

            self.client.indices.delete(index=index_name)
            return {"message": f"Index '{index_name}' deleted successfully."}
        except Exception as e:
            raise RuntimeError(f"Unexpected error during index deletion: {str(e)}")