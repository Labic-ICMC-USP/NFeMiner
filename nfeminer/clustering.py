
import warnings, inspect, igraph as ig
from typing import Any, Dict, Iterable, List, Tuple, Optional

class NFeCluster:
    """
    Orchestrator for clustering using a precomputed compact graph from an EdgeGenerator.

    The class is intentionally small: it asks the EdgeGenerator for a compact igraph.Graph
    (unique representatives only) at initialization, then exposes `run()` to execute a
    chosen igraph community detection algorithm and expands the compact labels back to
    the original IDs using `edge_generator.expand_clusters(...)`.

    Example:
        gen = StringMatchEdgeGenerator(items, ids, cache_dir)
        clusterer = NFeCluster(edge_generator=gen, threshold=0.85)

        # Run Louvain
        expanded = clusterer.run("louvain", resolution=1.0)
        # expanded is list[(original_id, params_with_cluster), ...] length == len(ids)

    Args:
        edge_generator: Instance implementing generate_compact_graph(threshold) and
                        expand_clusters(compact_labels, params).
        threshold: Threshold used to filter edges when generating the compact graph.
    """

    def __init__(self, edge_generator: Any, threshold: float = 0.0) -> None:
        # Basic validation
        if not hasattr(edge_generator, "generate_compact_graph") or not hasattr(
            edge_generator, "expand_clusters"
        ):
            raise ValueError(
                "edge_generator must implement generate_compact_graph(threshold) and expand_clusters(labels, params)"
            )

        self.edge_generator = edge_generator
        self.threshold = float(threshold)

        # create compact graph once and reuse for subsequent run() calls
        # compact_graph is an igraph.Graph of representative nodes only
        self.compact_graph: ig.Graph = self.edge_generator.generate_compact_graph(self.threshold)

    # -------------------------
    # Public API
    # -------------------------
    def run(self, algorithm: str, **kwargs) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Run the requested clustering algorithm on the compact graph, then expand results.

        Args:
            algorithm: name of the algorithm to run. Supported names:
                - "louvain"
                - "leiden"
                - "label_propagation"
                - "walktrap"
                - "fastgreedy"
                - "infomap"
                - "connected_components"
            **kwargs: extra parameters forwarded to the underlying igraph method when applicable.

        Returns:
            List[Tuple[original_id, params_with_cluster]]: list with length equal to the
            total number of original IDs; each entry contains the original ID and a dict
            with metadata + assigned cluster (the cluster value is assigned inside
            edge_generator.expand_clusters and will be a UUID per cluster).
        """
        algorithm_lower = algorithm.lower()

        # Run clustering on compact graph and obtain membership list
        membership = self._run_algorithm(self.compact_graph, algorithm_lower, **kwargs)

        # Build params dictionary to be stored for each original id (will be copied per id)
        params: Dict[str, Any] = {
            "algorithm": algorithm_lower,
            "algorithm_params": {k: v for k, v in kwargs.items()},
            "threshold": self.threshold
        }

        # Expand clusters using the edge generator (it will attach cluster UUID etc)
        expanded = self.edge_generator.expand_clusters(membership, params)
        return expanded

    # -------------------------
    # Internal helpers
    # -------------------------
    def _run_algorithm(self, graph: ig.Graph, algorithm: str, **kwargs) -> List[int]:
        """
        Execute an igraph community detection method and return a membership list
        (one cluster label per vertex of the compact graph).

        This function normalizes the variety of return types from igraph (VertexClustering
        objects, Clustering, VertexDendrogram, VertexSeq membership arrays, or ConnectedComponents).

        Args:
            graph: igraph.Graph to cluster (compact graph).
            algorithm: normalized name of the algorithm (lower case).
            **kwargs: forwarded to the igraph call when applicable.

        Returns:
            membership: List[int] of length graph.vcount() with a cluster label per vertex.
        """
        if graph.vcount() == 0:
            return []

        algo = algorithm.lower()

        # Dispatch table: map algorithm name -> callable that returns a clustering-like object
        # Each entry accepts (graph, **kwargs) and must return either:
        # - igraph.VertexClustering (preferred), or
        # - list/iterable of labels (length == n_vertices), or
        # - igraph.Clustering/VertexDendrogram which can be converted to membership.
        if algo == "louvain":
            # community_multilevel returns VertexClustering
            clustering_obj = graph.community_multilevel(**self._filter_kwargs(graph.community_multilevel, kwargs))
            return self._clustering_to_membership(clustering_obj)

        elif algo == "leiden":
            if not hasattr(graph, "community_leiden"):
                raise RuntimeError("igraph was built without community_leiden support on this environment.")
            clustering_obj = graph.community_leiden(**self._filter_kwargs(graph.community_leiden, kwargs))
            return self._clustering_to_membership(clustering_obj)

        elif algo == "label_propagation":
            clustering_obj = graph.community_label_propagation(**self._filter_kwargs(graph.community_label_propagation, kwargs))
            return self._clustering_to_membership(clustering_obj)

        elif algo == "walktrap":
            # walktrap returns a VertexDendrogram; convert to clustering with as_clustering()
            wd = graph.community_walktrap(**self._filter_kwargs(graph.community_walktrap, kwargs))
            clustering_obj = wd.as_clustering()
            return self._clustering_to_membership(clustering_obj)

        elif algo == "fastgreedy":
            fg = graph.community_fastgreedy(**self._filter_kwargs(graph.community_fastgreedy, kwargs))
            clustering_obj = fg.as_clustering()
            return self._clustering_to_membership(clustering_obj)

        elif algo == "infomap":
            # infomap returns VertexClustering in most builds; if it doesn't exist raise informative error
            if not hasattr(graph, "community_infomap"):
                raise RuntimeError("igraph was built without community_infomap support on this environment.")
            clustering_obj = graph.community_infomap(**self._filter_kwargs(graph.community_infomap, kwargs))
            return self._clustering_to_membership(clustering_obj)

        elif algo in ("connected_components", "components", "clusters"):
            # connected components: igraph.Graph.clusters() returns VertexClustering-like object
            comp = graph.clusters()
            return self._clustering_to_membership(comp)

        else:
            raise ValueError(f"Unknown clustering algorithm '{algorithm}'. Supported: "
                             "louvain, leiden, label_propagation, walktrap, fastgreedy, infomap, connected_components")

    @staticmethod
    def _clustering_to_membership(clustering_obj: Any) -> List[int]:
        """
        Normalize igraph clustering-like objects to a membership list.

        Accepts:
            - igraph VertexClustering (has .membership)
            - igraph Clustering / VertexDendrogram (convert via .membership or .as_clustering())
            - iterable/list of ints

        Returns:
            membership: list[int] length == number of vertices
        """
        # VertexClustering or similar: try .membership first
        if hasattr(clustering_obj, "membership"):
            return list(clustering_obj.membership)

        # VertexDendrogram has as_clustering()
        if hasattr(clustering_obj, "as_clustering"):
            try:
                cl = clustering_obj.as_clustering()
                if hasattr(cl, "membership"):
                    return list(cl.membership)
            except Exception:
                pass

        # If it's already an iterable of labels
        try:
            membership = list(clustering_obj)
            if all(isinstance(x, (int, bool)) or (isinstance(x, (str,)) and x.isdigit()) for x in membership):
                # coerce to int
                return [int(x) for x in membership]
            return membership
        except Exception:
            raise RuntimeError("Unable to normalize clustering result into membership list.")

    @staticmethod
    def _filter_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only kwargs that are accepted by the target igraph function.

        This avoids igraph raising TypeError for unexpected args forwarded from user.
        """
        try:
            sig = inspect.signature(func)
            accepted = set(sig.parameters.keys())
            return {k: v for k, v in kwargs.items() if k in accepted}
        except Exception:
            # fallback: pass everything if introspection fails
            return dict(kwargs)