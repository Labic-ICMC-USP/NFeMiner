from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from typing import Union, Any, Optional

class EdgeGenerator(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    @abstractmethod
    def generate_edges(self) -> pd.DataFrame:
        """
        Must return a DataFrame with columns:
        source, target, edge_type, similarity_score
        """
        pass

def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, column_name: str = 'id_nfe') -> nx.Graph:
    """Build a NetworkX Graph from node and edge pandas DataFrames.

    Each row in `nodes_df` becomes a node; the node identifier is taken from
    the column specified by `column_name`. Each row in `edges_df` becomes an
    edge and must contain `source` and `target` columns with node identifiers.
    All other columns from the rows are set as node/edge attributes.

    Args:
        nodes_df (pd.DataFrame): DataFrame where each row represents a node.
        edges_df (pd.DataFrame): DataFrame where each row represents an edge.
        column_name (str): Column name in `nodes_df` that contains the node id.
            Defaults to `'id_nfe'`.

    Returns:
        nx.Graph: A NetworkX graph with node and edge attributes populated
            from the DataFrame rows.

    Raises:
        KeyError: If `column_name` is missing in `nodes_df` or if `edges_df`
            does not contain the required `source` and `target` columns.
    """
    if column_name not in nodes_df.columns:
        raise KeyError(f"nodes_df must contain the '{column_name}' column.")
    if 'source' not in edges_df.columns or 'target' not in edges_df.columns:
        raise KeyError("edges_df must contain 'source' and 'target' columns.")

    G = nx.Graph()

    # Add nodes with attributes (exclude the id column from attributes)
    for _, row in nodes_df.iterrows():
        node_id = row[column_name]
        attrs = row.to_dict()
        attrs.pop(column_name, None)
        G.add_node(node_id, **attrs)

    # Add edges with attributes (exclude source/target from attributes)
    for _, row in edges_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        attrs = row.to_dict()
        attrs.pop("source", None)
        attrs.pop("target", None)
        G.add_edge(src, tgt, **attrs)

    return G

class BERTEmbeddingEdgeGenerator(EdgeGenerator):
    """Generate graph edges based on SentenceBERT semantic similarity.

    This edge generator encodes texts using a SentenceTransformer and
    produces an edge list with cosine similarity scores between distinct
    document pairs.

    Attributes:
        model (SentenceTransformer): SentenceTransformer instance used to encode texts.
        df (pd.DataFrame): Input DataFrame provided by the base class.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model: Union[str, SentenceTransformer] = 'sentence-transformers/all-MiniLM-L6-v2',
    ) -> None:
        """Initialize the edge generator.

        Args:
            df (pd.DataFrame): Input DataFrame. Must contain a column matching
                `ID_NFE_COLUMN` (passed in generate_edges) and the text column you will pass to `generate_edges`.
            model (str | SentenceTransformer): If a string, the corresponding
                SentenceTransformer model is loaded; if an instance is provided,
                it is used directly. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        """
        super().__init__(df)
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
        else:
            self.model = model

    def generate_edges(self, field_text: str = 'text',ID_NFE_COLUMN:str='id_nfe') -> pd.DataFrame:
        """Compute pairwise semantic similarity and return edges with scores.

        For every pair of distinct rows (i, j) with i < j, this function
        computes the cosine similarity between their SentenceBERT embeddings
        and returns a DataFrame with columns:
            ['source', 'target', 'edge_type', 'similarity_score']

        Args:
            field_text (str): Name of the column in `self.df` containing text.
                Defaults to 'text'.
            ID_NFE_COLUMN (string,optional): Defines the column of id nfe. By default is 'id_nfe'

        Returns:
            pd.DataFrame: Edge list with similarity scores.

        Raises:
            KeyError: If `ID_NFE_COLUMN` or `field_text` is missing in `self.df`.
        """
        if ID_NFE_COLUMN not in self.df.columns:
            raise KeyError(f"DataFrame must contain '{ID_NFE_COLUMN}' column.")
        if field_text not in self.df.columns:
            raise KeyError(f"DataFrame must contain '{field_text}' column.")

        texts = self.df[field_text].fillna("").tolist()
        ids = self.df[ID_NFE_COLUMN].tolist()

        # Encode texts to embeddings (tensor). Let SentenceTransformer handle device placement.
        embeddings = self.model.encode(texts, convert_to_tensor=True)

        # Compute pairwise cosine similarity matrix (torch tensor)
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

        edges = []
        num_rows = len(ids)
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                score = float(sim_matrix[i][j].item())
                edges.append({
                    'source': ids[i],
                    'target': ids[j],
                    'edge_type': 'bert-similarity',
                    'similarity_score': score
                })

        return pd.DataFrame(edges)

class NCMSimilarityEdgeGenerator(EdgeGenerator):
    """Generate edges between records that share the same NCM value.

    This edge generator creates an undirected edge with similarity score 1
    for every distinct pair of rows in the DataFrame that have identical
    values in the specified NCM column.
    """
    
    def generate_edges(self, field_ncm: str = 'ncm',ID_NFE_COLUMN:str='id_nfe') -> pd.DataFrame:
        """Create edges between rows with equal NCM codes.

        For each group of rows sharing the same `field_ncm` value, this method
        emits an edge for every unique pair (i, j) with i < j.
        
        Args:
            field_ncm (str): Column name containing the NCM code. Defaults to 'ncm'.
            ID_NFE_COLUMN (string,optional): Defines the column of id nfe. By default is 'id_nfe'
        Returns:
            pd.DataFrame: DataFrame with columns ['source', 'target', 'edge_type', 'similarity_score'].
        """
        grouped = self.df.groupby(field_ncm)
        edges = []
        for ncm, subdf in grouped:
            for i, (iidx, row_i) in enumerate(subdf.iterrows()):
                for j, (jidx, row_j) in enumerate(subdf.iterrows()):
                    if i >= j:
                        continue

                    edges.append({
                        "source": row_i[ID_NFE_COLUMN],
                        "target": row_j[ID_NFE_COLUMN],
                        "edge_type": "ncm_similarity",
                        "similarity_score": 1
                    })
        return pd.DataFrame(edges)

class ValueRangeEdgeGenerator(EdgeGenerator):
    """Generate edges for rows whose numeric field falls within a given range.

    Edges are produced only between rows that:
      * belong to the same group (field_group), and
      * have the `field_valor` value inside (min_value, max_value).

    The produced edge's `similarity_score` is the (possibly normalized) value
    taken from the `field_valor` column.

    Note:
        This implementation emits an edge for each unique pair (i, j) with i < j
        inside each group (no cross-group edges).
    """
    
    def generate_edges(
        self,
        min_value: Optional[float],
        max_value: Optional[float],
        normalize: bool = False,
        field_valor: str = 'valor',
        field_group: str = 'unidade',
        ID_NFE_COLUMN:str='id_nfe'
    ) -> pd.DataFrame:
        """Create edges for records with `field_valor` inside [min_value, max_value].

        Args:
            min_value (float | None): Lower bound for `field_valor`. If None, the
                minimum value from `self.df[field_valor]` is used.
            max_value (float | None): Upper bound for `field_valor`. If None, the
                maximum value from `self.df[field_valor]` is used.
            normalize (bool): If True, the similarity score is normalized to [0, 1]
                using (value - min_value) / (max_value - min_value).
            field_valor (str): Column name that contains the numeric value.
            field_group (str): Column name used to group rows (edges only inside group).

        Returns:
            pd.DataFrame: DataFrame with columns ['source', 'target', 'edge_type', 'similarity_score'].

        Raises:
            KeyError: If required columns are missing from the DataFrame.
        """
        # Basic validations
        if field_valor not in self.df.columns:
            raise KeyError(f"DataFrame must contain column '{field_valor}'")
        if field_group not in self.df.columns:
            raise KeyError(f"DataFrame must contain column '{field_group}'")
        if ID_NFE_COLUMN not in self.df.columns:
            raise KeyError(f"DataFrame must contain column '{ID_NFE_COLUMN}'")

        # Resolve min/max if not provided
        if min_value is None:
            min_value = float(self.df[field_valor].min())
        if max_value is None:
            max_value = float(self.df[field_valor].max())

        grouped = self.df.groupby(field_group)
        edges = []
        for group_val, subdf in grouped:
            for i, (iidx, row_i) in enumerate(subdf.iterrows()):
                for j, (jidx, row_j) in enumerate(subdf.iterrows()):
                    if i >= j:
                        continue

                    sim = row_i[field_valor]
                    # check range (open interval as in original: min_value < sim < max_value)
                    if not (min_value < sim and sim < max_value):
                        continue

                    if normalize:
                        denom = (max_value - min_value)
                        # avoid division by zero
                        if denom == 0:
                            sim = 0.0
                        else:
                            sim = (sim - min_value) / denom

                    edges.append({
                        "source": row_i[ID_NFE_COLUMN],
                        "target": row_j[ID_NFE_COLUMN],
                        "edge_type": "value_range",
                        "similarity_score": sim
                    })

        return pd.DataFrame(edges)

class StringMatchEdgeGenerator(EdgeGenerator):
    """Generate edges between records whose text fields are nearly identical.

    This generator uses difflib.SequenceMatcher to compute a similarity ratio
    between two strings and emits an edge when the ratio exceeds a threshold 
    (0.95 by default).
    """
    
    def generate_edges(self, field_str: str = 'descricao',threshold=0.95,ID_NFE_COLUMN='id_nfe') -> pd.DataFrame:
        """Create edges for highly similar string pairs.

        For every unique pair of rows (i, j) with i < j, this method computes
        the SequenceMatcher ratio between the values in `field_str`. If the
        similarity ratio is greater than threshold, an undirected edge is emitted
        with `edge_type` == "string_match" and `similarity_score` equal to the ratio.

        Args:
            field_str (str): Column name containing the text to compare.
                Defaults to 'descricao'.
            threshold (float): Determines the minimal similarity to construct an edge. By 
                default, it's 0.95.
            ID_NFE_COLUMN (string,optional): Defines the column of id nfe. By default is 'id_nfe'

        Returns:
            pd.DataFrame: DataFrame with columns ['source', 'target', 'edge_type', 'similarity_score'].
        """
        edges = []
        for i, row_i in self.df.iterrows():
            for j, row_j in self.df.iterrows():
                if i >= j:
                    continue
                sim = SequenceMatcher(None, row_i[field_str], row_j[field_str]).ratio()
                if sim > threshold:
                    edges.append({
                        "source": row_i[ID_NFE_COLUMN],
                        "target": row_j[ID_NFE_COLUMN],
                        "edge_type": "string_match",
                        "similarity_score": sim
                    })
        return pd.DataFrame(edges)

class PriceBandEdgeGenerator(EdgeGenerator):
    """
    Edge generator based on the 'valor' field (price or numeric attribute). 
    
    This generator creates edges between nodes only if their associated 'valor'
    lies within a specified numeric band (`min_value`, `max_value`). 
    The `similarity_score` of each edge corresponds to the value itself and can 
    optionally be normalized to the [0,1] range, based on the given bounds.
    
    Typical use case: grouping or connecting nodes that fall into the same price
    range or value interval.
    """
        
    def generate_edges(self,
                       min_value: float,
                       max_value: float,
                       normalize: bool = False,
                       field_valor: str = 'valor',
                       ID_NFE_COLUMN='id_nfe') -> pd.DataFrame:
        """
        Generate edges between nodes based on whether their value field 
        falls within a specified price band.

        Edges are generated only if the field value is within the specified
        'min_value' and 'max_value' range. The 'similarity_score' is equal 
        to the node's value and can be normalized.

        Args:
            min_value (float): Minimum value of the 'valor' field. If None, 
                the minimum found in the dataset is used.
            max_value (float): Maximum value of the 'valor' field. If None, 
                the maximum found in the dataset is used.
            normalize (bool): If True, applies normalization to the 
                similarity score. Normalization maps:
                    - 'min_value' (if not None) → 0
                    - 'max_value' (if not None) → 1
            field_valor (str): Column name containing the value field.
            ID_NFE_COLUMN (string,optional): Defines the column of id nfe. By default is 'id_nfe'

        Returns:
            pd.DataFrame: A DataFrame containing the generated edges, 
            with columns ['source', 'target', 'edge_type', 'similarity_score'].
        """
                
        if min_value is None:
            min_value = min(self.df[field_valor])
        if max_value is None:
            max_value = max(self.df[field_valor])
            
        edges = []
        for i, (iidx, row_i) in enumerate(self.df.iterrows()):
            for j, (jidx, row_j) in enumerate(self.df.iterrows()):
                
                if i >= j:
                    continue
                
                sim = row_i[field_valor]
                if not (min_value < sim and sim < max_value):
                    continue
                
                if normalize:
                    sim = (sim - min_value) / (max_value - min_value)
                
                edges.append({
                    "source": row_i[ID_NFE_COLUMN],
                    "target": row_j[ID_NFE_COLUMN],
                    "edge_type": "price_band",
                    "similarity_score": sim
                })
        return pd.DataFrame(edges)
