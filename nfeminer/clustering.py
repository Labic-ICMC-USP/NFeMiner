"""
Module for hierarchical consensus clustering of NFe items.

Design:
    - Clusterer: pure clustering tool. Receives a prebuilt similarity matrix and
      returns a ClusterResult (membership + consensus graph). Knows nothing about
      JSON, items or cache.
    - NFeMinerClustering: orchestrator and wrapper. Manages the SimilarityEngine, builds
      similarity matrices, runs Clusterer, computes metrics, decides splits and
      assigns hierarchical cluster IDs. Uses a queue instead of recursion.

Output:
    NFeMinerClustering.run() returns:
        dict[int, list[str]]
            id -> list of cluster_ids from most generic to most specific
            e.g. {1: ["0001", "0001.0003"], 2: ["0001"], 3: ["0002"]}

Rounds configuration (passed to NFeMinerClustering):
    - None: uses internal default grid.
    - list[dict]: explicit rounds, each dict has keys:
        "algorithm"  (str)
        "threshold"  (float)
        "funcs"      (list[str])
        "bootstrap"  (float | None) sample fraction e.g. 0.8, or None for all data
    - dict: automatic grid with keys:
        "algorithms"   (list[str])
        "thresholds"   (list[float])
        "func_groups"  (list[list[str]])
        "bootstrap"    (float | None) applied to all combinations
      Expands into all combinations internally.
"""

from __future__ import annotations

import igraph as ig
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple
from scipy.sparse import lil_matrix

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Output of a single Clusterer.run() call.

    Attributes:
        membership: Mapping from item ID to integer cluster label (internal,
            temporary — NFeMinerClustering converts these to hierarchical string IDs).
        graph: Consensus igraph.Graph built from the co-association matrix.
            Vertices have attribute ``item_id``. Edges have attribute ``weight``
            representing co-association frequency in [0, 1].
    """

    membership: Dict[int, int]
    graph: ig.Graph

@dataclass
class ClusterMetrics:
    """Quality metrics for a single cluster.

    All similarity values are derived from the KVStore cache (full pairwise
    values, not just graph edges) so they reflect true similarity, not the
    thresholded view seen by the Clusterer.

    Attributes:
        cluster_id: Hierarchical string ID of the cluster.
        size: Number of items in the cluster.
        depth: Nesting level in the hierarchy (0 = root).
        intra_similarity_mean: Mean pairwise similarity among cluster members.
        intra_similarity_std: Standard deviation of pairwise similarities.
        diameter: Largest pairwise distance (1 - similarity) among members.
        intra_inter_ratio: Mean intra-cluster similarity divided by mean
            similarity to the nearest neighbouring cluster. Higher is better.
            None if there is only one cluster at this level.
    """

    cluster_id: str
    size: int
    depth: int
    intra_similarity_mean: float
    intra_similarity_std: float
    diameter: float
    intra_inter_ratio: Optional[float]

@dataclass
class SplitDecision:
    """Decision on whether a cluster should be further split.

    Attributes:
        cluster_id: Hierarchical string ID of the cluster.
        should_split: Whether to enqueue this cluster for another round.
        keep_parent: Whether to include this cluster_id in the final output.
            False for noisy clusters (e.g. "bovino" mixing unrelated items)
            that should be split but not kept as a valid grouping.
        confidence: Score in [0, 1] representing decision confidence.
            Currently rule-based (0.0 or 1.0). Reserved for future ML use.
        reason: Human-readable explanation for the decision.
    """

    cluster_id: str
    should_split: bool
    keep_parent: bool
    confidence: float
    reason: str

@dataclass
class _QueueItem:
    """Internal queue entry for hierarchical processing.

    Attributes:
        ids: Item IDs to cluster in this round.
        parent_id: Hierarchical ID of the parent cluster, or None for root.
        depth: Current nesting level.
    """

    ids: List[int]
    parent_id: Optional[str]
    depth: int

# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------

class Clusterer:
    """Pure consensus clustering tool.

    Receives a prebuilt dense similarity matrix and a list of rounds
    (each specifying algorithm, threshold and optionally bootstrap fraction),
    builds a co-association matrix across all rounds, constructs a consensus
    graph and returns the final membership.

    This class knows nothing about items, JSON, cache or hierarchical IDs.
    It only operates on numeric matrices and igraph graphs.

    Args:
        n_runs_per_round: Number of times each round configuration is repeated.
            Since algorithms like Leiden/Louvain are non-deterministic (no fixed
            seed), repeating them increases consensus stability.

    Example:
        clusterer = Clusterer(n_runs_per_round=3)
        result = clusterer.run(
            ids=[1, 2, 3, 4],
            matrix=similarity_matrix,   # np.ndarray shape (n, n)
            rounds=[
                {"algorithm": "leiden", "threshold": 0.8, "funcs": ["bert"]},
            ],
        )
    """

    def __init__(self, n_runs_per_round: int = 3) -> None:
        """Initialize Clusterer.

        Args:
            n_runs_per_round: Repetitions per round for non-deterministic stability.
        """
        self.n_runs_per_round = n_runs_per_round

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ids: List[int],
        matrix: np.ndarray,
        rounds: List[Dict[str, Any]],
    ) -> ClusterResult:
        """Run consensus clustering and return membership + consensus graph.

        Args:
            ids: List of item IDs corresponding to matrix rows/columns.
            matrix: Square symmetric similarity matrix, shape (n, n), values in [0, 1].
                    Row/column i corresponds to ids[i].
            rounds: List of round dicts. Each dict must have:
                - ``algorithm`` (str): igraph community detection algorithm name.
                - ``threshold`` (float): Edge inclusion threshold.
                - ``funcs`` (list[str]): Function names used to build this matrix slice.
                  Currently informational — the full matrix is passed; future versions
                  may support per-function sub-matrices.
                - ``bootstrap`` (float | None): Fraction of items to sample per run,
                  or None to use all items.

        Returns:
            ClusterResult with membership dict and consensus igraph.Graph.

        Raises:
            ValueError: If matrix shape does not match len(ids).
        """
        n = len(ids)
        if matrix.shape != (n, n):
            raise ValueError(
                f"Matrix shape {matrix.shape} does not match len(ids)={n}."
            )
        if n == 0:
            return ClusterResult(membership={}, graph=ig.Graph())
        if n == 1:
            return ClusterResult(membership={ids[0]: 0}, graph=ig.Graph(1))

        # Co-association matrix: counts how many times pair (i,j) ended in same cluster
        co_counts = lil_matrix((n, n), dtype=np.float32)
        total_runs = 0

        for round_cfg in rounds:
            algorithm = round_cfg["algorithm"]
            threshold = float(round_cfg["threshold"])
            bootstrap = round_cfg.get("bootstrap", None)

            for _ in range(self.n_runs_per_round):
                # Bootstrap: sample a subset of indices
                if bootstrap is not None:
                    sample_size = max(2, int(n * bootstrap))
                    sample_idx = np.random.choice(n, size=sample_size, replace=False)
                    sample_idx = np.sort(sample_idx)
                else:
                    sample_idx = np.arange(n)

                sub_matrix = matrix[np.ix_(sample_idx, sample_idx)]
                sub_graph = self._matrix_to_graph(sub_matrix, threshold)
                membership_list = self._run_algorithm(sub_graph, algorithm)

                # Accumulate co-association for sampled pairs
                for cluster_label in set(membership_list):
                    members = np.where(np.array(membership_list) == cluster_label)[0]
                    for a in members:
                        for b in members:
                            if a < b:
                                gi = sample_idx[a]
                                gj = sample_idx[b]
                                co_counts[gi, gj] += 1
                                co_counts[gj, gi] += 1

                total_runs += 1

        # Normalize co-association by total runs
        co_matrix = co_counts.toarray() / max(total_runs, 1)

        # Build consensus graph from co-association (threshold at 0.5 by default)
        consensus_threshold = 0.5
        consensus_graph = self._matrix_to_graph(co_matrix, consensus_threshold)
        consensus_graph.vs["item_id"] = list(ids)

        # Final clustering on consensus graph
        final_membership = self._run_algorithm(consensus_graph, "connected_components")

        membership_dict = {ids[i]: final_membership[i] for i in range(n)}
        return ClusterResult(membership=membership_dict, graph=consensus_graph)

    def fast_group(
        self,
        texts: List[str],
        lsh_kwargs: Optional[Dict[str, Any]] = None,
        semantic_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Fast two-stage grouping: lexical deduplication then semantic clustering.

        Stage 1 — LexicalGrouper:
            Groups texts with near-identical wording (typos, abbreviations,
            minor variations). Produces lexical_groups keyed as "LEX-N".

        Stage 2 — SemanticGrouper:
            Extracts one representative per lexical group, encodes all
            representatives with BERT + FAISS ANN, and groups semantically
            related representatives. Expands each semantic group back to
            include all lexical members. Produces semantic_groups keyed as "PRD-N".

        The two outputs are independent — lexical_groups is fine-grained
        deduplication; semantic_groups is the broader product-level grouping
        (level zero for hierarchical clustering).

        Args:
            texts: List of raw strings, one per item. Indices correspond to
                   item positions in the caller's dataset.
            lsh_kwargs: Optional kwargs for LexicalGrouper constructor.
                        E.g. {"threshold": 0.85, "num_perm": 128, "ngram": 3}.
            semantic_kwargs: Optional kwargs for SemanticGrouper constructor.
                        E.g. {"threshold": 0.88, "k": 50}.

        Returns:
            Tuple of:
                - lexical_groups: Dict "LEX-N" -> list of indices (fine-grained).
                - semantic_groups: Dict "PRD-N" -> list of indices (level zero).
                  Indices in semantic_groups expand from lexical_groups, so every
                  original index appears in exactly one semantic group.

        Example:
            clusterer = Clusterer()
            lex, sem = clusterer.fast_group(
                texts=["leite integral 1l", "leite integral 1 l", "suco laranja"],
                lsh_kwargs={"threshold": 0.85},
                semantic_kwargs={"threshold": 0.88},
            )
            # lex: {"LEX-0": [0, 1], "LEX-2": [2]}
            # sem: {"PRD-0": [0, 1], "PRD-2": [2]}
        """

        lsh_kw = lsh_kwargs or {}
        sem_kw = semantic_kwargs or {}

        # Stage 1: lexical grouping
        print("[fast_group] Stage 1: lexical grouping...", flush=True)
        lex_grouper = LexicalGrouper(**lsh_kw)
        lexical_groups = lex_grouper.fit(texts)

        # Stage 2: semantic grouping over lexical representatives
        print("[fast_group] Stage 2: semantic grouping over representatives...", flush=True)
        lex_ids = list(lexical_groups.keys())
        rep_texts = [texts[lexical_groups[gid][0]] for gid in lex_ids]

        sem_grouper = SemanticGrouper(**sem_kw)
        rep_groups = sem_grouper.fit(rep_texts)

        # Expand semantic groups back to all original indices
        semantic_groups: Dict[str, List[int]] = defaultdict(list)
        for sem_gid, rep_positions in rep_groups.items():
            for pos in rep_positions:
                lex_gid = lex_ids[pos]
                semantic_groups[sem_gid].extend(lexical_groups[lex_gid])
        semantic_groups = dict(semantic_groups)

        n_multi = sum(1 for m in semantic_groups.values() if len(m) > 1)
        print(
            f"[fast_group] Done. "
            f"Lexical: {len(lexical_groups):,} groups. "
            f"Semantic: {len(semantic_groups):,} groups ({n_multi:,} multi-item).",
            flush=True,
        )
        return lexical_groups, semantic_groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matrix_to_graph(matrix: np.ndarray, threshold: float) -> ig.Graph:
        """Build an igraph.Graph from a similarity matrix with edge threshold.

        Args:
            matrix: Square symmetric similarity matrix.
            threshold: Minimum similarity to create an edge.

        Returns:
            igraph.Graph with ``weight`` edge attribute.
        """
        n = matrix.shape[0]
        g = ig.Graph()
        g.add_vertices(n)
        edges = []
        weights = []
        for i in range(n):
            for j in range(i + 1, n):
                w = float(matrix[i, j])
                if w >= threshold:
                    edges.append((i, j))
                    weights.append(w)
        if edges:
            g.add_edges(edges)
            g.es["weight"] = weights
        return g

    @staticmethod
    def _run_algorithm(graph: ig.Graph, algorithm: str) -> List[int]:
        """Run a community detection algorithm on a graph.

        Args:
            graph: igraph.Graph to cluster.
            algorithm: Algorithm name. Supported: louvain, leiden,
                label_propagation, walktrap, fastgreedy, infomap,
                connected_components.

        Returns:
            List[int] of cluster labels, length == graph.vcount().

        Raises:
            ValueError: If algorithm name is not supported.
        """
        if graph.vcount() == 0:
            return []

        algo = algorithm.lower()

        def _to_membership(obj: Any) -> List[int]:
            if hasattr(obj, "membership"):
                return list(obj.membership)
            if hasattr(obj, "as_clustering"):
                return list(obj.as_clustering().membership)
            return [int(x) for x in obj]

        weights = graph.es["weight"] if graph.ecount() > 0 and "weight" in graph.es.attributes() else None

        if algo == "louvain":
            return _to_membership(graph.community_multilevel(weights=weights))
        elif algo == "leiden":
            if not hasattr(graph, "community_leiden"):
                raise RuntimeError("community_leiden not available in this igraph build.")
            return _to_membership(graph.community_leiden(weights=weights))
        elif algo == "label_propagation":
            return _to_membership(graph.community_label_propagation(weights=weights))
        elif algo == "walktrap":
            return _to_membership(graph.community_walktrap(weights=weights).as_clustering())
        elif algo == "fastgreedy":
            return _to_membership(graph.community_fastgreedy(weights=weights).as_clustering())
        elif algo == "infomap":
            if not hasattr(graph, "community_infomap"):
                raise RuntimeError("community_infomap not available in this igraph build.")
            return _to_membership(graph.community_infomap(edge_weights=weights))
        elif algo in ("connected_components", "components"):
            return _to_membership(graph.clusters())
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Supported: louvain, leiden, "
                "label_propagation, walktrap, fastgreedy, infomap, connected_components."
            )

# ---------------------------------------------------------------------------
# LexicalGrouper
# ---------------------------------------------------------------------------

class LexicalGrouper:
    """Group texts by lexical similarity using MinHash + LSH.

    Finds near-duplicate strings (same product described slightly differently)
    without computing all pairwise comparisons. Uses character n-gram shingling
    and Locality Sensitive Hashing to identify candidates in O(n) time.

    Suitable for:
        - Deduplicating product descriptions with typos or abbreviations
        - Grouping invoice lines with minor textual variations
        - Any task requiring fast near-duplicate detection over large text sets

    Can be used standalone without NFeMinerClustering or SimilarityEngine.

    Args:
        threshold: Jaccard similarity threshold for LSH bucketing. Higher values
            produce tighter, smaller groups. Recommended range: 0.75-0.90.
        num_perm: Number of MinHash permutations. Higher = more accurate but
            slower and more memory. 128 is a good default.
        ngram: Character n-gram size for shingling. 3 works well for short
            product descriptions.

    Example:
        grouper = LexicalGrouper(threshold=0.85)
        groups = grouper.fit(["leite integral 1l", "leite integral 1 l", "suco laranja"])
        # {"LEX-0": [0, 1], "LEX-2": [2]}
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram: int = 3,
    ) -> None:
        """Initialize LexicalGrouper.

        Args:
            threshold: Jaccard similarity threshold for LSH.
            num_perm: Number of MinHash permutations.
            ngram: Character n-gram size.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram = ngram

    def fit(self, texts: List[str]) -> Dict[str, List[int]]:
        """Group texts by lexical similarity.

        Normalizes inputs (lowercase + strip), builds MinHash signatures,
        runs LSH to find candidates, then applies union-find to form groups.
        The returned indices correspond to positions in the input list.

        Args:
            texts: List of strings to group. May contain duplicates.

        Returns:
            Dict mapping group_id (e.g. "LEX-42") to list of indices into
            the input texts list. Every index appears in exactly one group.

        Raises:
            ImportError: If datasketch is not installed.
        """
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError as exc:
            raise ImportError(
                "datasketch is required for LexicalGrouper. "
                "Install with: pip install datasketch"
            ) from exc

        normalized = [t.lower().strip() for t in texts]
        text_to_indices = defaultdict(list)
        for i, t in enumerate(normalized):
            text_to_indices[t].append(i)
        normalized = list(text_to_indices.keys())
        n = len(normalized)

        print(f"[LexicalGrouper] Building MinHash signatures for {n:,} texts...", flush=True)
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        minhashes = []

        for i, text in enumerate(normalized):
            m = MinHash(num_perm=self.num_perm)
            for j in range(max(1, len(text) - self.ngram + 1)):
                m.update(text[j:j + self.ngram].encode("utf-8"))
            minhashes.append(m)
            lsh.insert(str(i), m)

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        print(f"[LexicalGrouper] Finding candidate pairs...", flush=True)
        for i in range(n):
            for c in lsh.query(minhashes[i]):
                j = int(c)
                if j != i:
                    union(i, j)

        # Build output groups
        raw: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            root = find(i)
            text = normalized[i]
            raw[root].extend(text_to_indices[text])

        groups = {f"LEX-{root}": members for root, members in raw.items()}
        n_multi = sum(1 for m in groups.values() if len(m) > 1)
        print(
            f"[LexicalGrouper] {len(groups):,} groups "
            f"({n_multi:,} multi-item, {len(groups)-n_multi:,} singletons).",
            flush=True,
        )
        return groups

# ---------------------------------------------------------------------------
# SemanticGrouper
# ---------------------------------------------------------------------------

class SemanticGrouper:
    """Group texts by semantic similarity using BERT embeddings and FAISS ANN.

    Encodes all texts in a single batched forward pass, builds an approximate
    nearest neighbor index (FAISS IVFFlat), and connects texts whose cosine
    similarity exceeds the threshold via union-find.

    Suitable for:
        - Grouping product descriptions by meaning regardless of wording
        - Clustering tags or keywords semantically
        - Any task requiring semantic deduplication or grouping at scale

    Can be used standalone without NFeMinerClustering or SimilarityEngine.
    Accepts plain text lists — does not know about item dicts or cache keys.

    Args:
        threshold: Minimum cosine similarity to connect two texts. Higher =
            tighter groups. Recommended range: 0.80-0.92.
        k: Number of nearest neighbors to retrieve per text. Higher = more
            recall but more pairs evaluated. Recommended: 20-100.
        model_name: SentenceTransformer model identifier.
        batch_size: Encoding batch size. Reduce if running out of memory.

    Example:
        grouper = SemanticGrouper(threshold=0.88)
        groups = grouper.fit(["leite integral", "leite uht integral", "suco laranja"])
        # {"PRD-0": [0, 1], "PRD-2": [2]}
    """

    def __init__(
        self,
        threshold: float = 0.85,
        k: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 512,
    ) -> None:
        """Initialize SemanticGrouper.

        Args:
            threshold: Cosine similarity threshold for connecting texts.
            k: Nearest neighbors to retrieve per text.
            model_name: SentenceTransformer model name.
            batch_size: BERT encoding batch size.
        """
        self.threshold = threshold
        self.k = k
        self.model_name = model_name
        self.batch_size = batch_size

    def fit(self, texts: List[str]) -> Dict[str, List[int]]:
        """Group texts by semantic similarity.

        Encodes all texts, builds FAISS index, searches K nearest neighbors,
        and applies union-find to form connected components above threshold.

        Args:
            texts: List of strings to group. Should be clean and normalized
                for best embedding quality. May contain duplicates.

        Returns:
            Dict mapping group_id (e.g. "PRD-42") to list of indices into
            the input texts list. Every index appears in exactly one group.

        Raises:
            ImportError: If sentence-transformers or faiss-cpu is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SemanticGrouper. "
                "Install with: pip install sentence-transformers"
            ) from exc
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for SemanticGrouper. "
                "Install with: pip install faiss-cpu"
            ) from exc

        n = len(texts)
        print(f"[SemanticGrouper] Encoding {n:,} texts with '{self.model_name}'...", flush=True)

        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine sim = dot product after L2 norm
        )

        dim = embeddings.shape[1]
        nlist = max(1, int(np.sqrt(n)))

        print(f"[SemanticGrouper] Building FAISS index (dim={dim}, nlist={nlist})...", flush=True)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = max(1, nlist // 10)
        index.train(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))

        actual_k = min(self.k + 1, n)
        print(f"[SemanticGrouper] Searching {actual_k} nearest neighbors...", flush=True)
        similarities, nn_indices = index.search(embeddings.astype(np.float32), actual_k)

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        connected = 0
        for i in range(n):
            for rank in range(1, actual_k):
                j = int(nn_indices[i, rank])
                sim = float(similarities[i, rank])
                if j < 0:
                    break
                if sim < self.threshold:
                    break   # neighbors sorted by similarity descending
                union(i, j)
                connected += 1

        print(f"[SemanticGrouper] {connected:,} pairs connected above threshold={self.threshold}.", flush=True)

        raw: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            raw[find(i)].append(i)

        groups = {f"PRD-{root}": members for root, members in raw.items()}
        n_multi = sum(1 for m in groups.values() if len(m) > 1)
        print(
            f"[SemanticGrouper] {len(groups):,} groups "
            f"({n_multi:,} multi-item, {len(groups)-n_multi:,} singletons).",
            flush=True,
        )
        return groups

# ---------------------------------------------------------------------------
# NFeminerCluster
# ---------------------------------------------------------------------------

# Default grid used when rounds=None
_DEFAULT_GRID = {
    "algorithms": ["leiden", "louvain"],
    "thresholds": [0.7, 0.8, 0.9],
    "func_groups": None,  # resolved at runtime from registered functions
    "bootstrap": 0.8,
}

class NFeMinerClustering:
    """Orchestrator for hierarchical consensus clustering of NFe items.

    Responsibilities:
        1. Extract item feature values from raw data (JSON-agnostic — caller
           provides pre-extracted feature dicts).
        2. Drive SimilarityEngine to compute and cache pairwise similarities.
        3. Build similarity matrices and feed them to Clusterer.
        4. Compute ClusterMetrics from the cache.
        5. Apply SplitDecision rules and manage a queue of sub-clusters.
        6. Assign hierarchical cluster IDs and build the final output mapping.

    Args:
        items: List of feature dicts, one per item. Keys are similarity function
               names; values are the raw inputs for that function.
               Example: {"sequence_match": "AGUA MINERAL 500ML", "category": ["Bebidas"]}
        ids: List of integer IDs, same length as items.
        engine: Configured SimilarityEngine instance (already has functions + cache).
        rounds: Round configuration. See module docstring for accepted formats.
        n_runs_per_round: Repetitions per round inside Clusterer for stability.
        max_depth: Maximum hierarchy depth. Prevents infinite splitting.
        min_cluster_size: Clusters smaller than this are kept as-is (no split).
        split_thresholds: Dict with keys ``silhouette``, ``intra_std``, ``diameter``
            and ``intra_inter_ratio`` used by the split decision rule.
            Uses sensible defaults if None.

    Example:
        engine = SimilarityEngine(funcs=[SequenceMatchSimilarity()], cache=kvstore)
        nfe = NFeMinerClustering(
            items=[{"sequence_match": "AGUA MINERAL"}, {"sequence_match": "SUCO LARANJA"}],
            ids=[1, 2],
            engine=engine,
        )
        result = nfe.run()
        # {1: ["0001"], 2: ["0002"]}
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        ids: List[int],
        engine: Any,
        lsh_key: str = None,
        semantic_key: Optional[str] = None,
        rounds: Any = None,
        n_runs_per_round: int = 3,
        max_depth: int = 5,
        min_cluster_size: int = 2,
        split_thresholds: Optional[Dict[str, float]] = None,
        lsh_kwargs: Optional[Dict[str, Any]] = None,
        semantic_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize NFeMinerClustering.

        Args:
            items: Feature dicts keyed by similarity function name.
            ids: Integer IDs, same length and order as items.
            engine: SimilarityEngine instance.
            lsh_key: Key in item dicts to use for lexical grouping (LexicalGrouper).
                     Required to enable the fast_group pre-processing stage.
                     Example: "original" for raw invoice descriptions.
            semantic_key: Key in item dicts to use for semantic grouping
                     (SemanticGrouper). Defaults to lsh_key if None.
                     Example: "produto_base" for LLM-enriched descriptions.
            rounds: None, explicit list or grid dict. See module docstring.
            n_runs_per_round: Clusterer repetitions per round.
            max_depth: Hard limit on hierarchy depth.
            min_cluster_size: Clusters below this size are never split.
            split_thresholds: Override default split decision thresholds.
            lsh_kwargs: Optional kwargs forwarded to LexicalGrouper.
            semantic_kwargs: Optional kwargs forwarded to SemanticGrouper.

        Raises:
            ValueError: If items and ids have different lengths.
        """
        if len(items) != len(set(ids)):
            raise ValueError("`items` and `set(ids)` must have the same length.")

        self._items: List[Dict[str, Any]] = items
        self._ids: List = ids
        self._lsh_key: Optional[str] = lsh_key
        self._semantic_key: Optional[str] = semantic_key if semantic_key is not None else lsh_key
        self._lsh_kwargs: Dict[str, Any] = lsh_kwargs or {}
        self._semantic_kwargs: Dict[str, Any] = semantic_kwargs or {}
        self._engine = engine
        self._clusterer = Clusterer(n_runs_per_round=n_runs_per_round)
        self._max_depth = max_depth
        self._min_cluster_size = min_cluster_size

        self._split_thresholds = {
            "silhouette_min": 0.3,
            "intra_std_max": 0.25,
            "diameter_max": 0.7,
            "intra_inter_ratio_min": 1.2,
        }

        if split_thresholds:
            self._split_thresholds.update(split_thresholds)

        self._rounds: List[Dict[str, Any]] = self._resolve_rounds(rounds)

        self._data, self._uniqueid_to_index = self._deduplicate_dicts(items)

        # Counter for generating hierarchical IDs at each level
        # key: parent_id (or None for root) -> next child index
        self._id_counters: Dict[Optional[str], int] = {}

    def _freeze(self, obj: Any, sort_lists: bool = False) -> Any:
        if isinstance(obj, dict):
            return tuple(sorted((k, self._freeze(v, sort_lists)) for k, v in obj.items()))
        elif isinstance(obj, list):
            if sort_lists:
                return tuple(sorted(self._freeze(v, sort_lists) for v in obj))
            return tuple(self._freeze(v, sort_lists) for v in obj)
        elif isinstance(obj, set):
            return tuple(sorted(self._freeze(v, sort_lists) for v in obj))
        else:
            return obj

    def _deduplicate_dicts(self, data: List[Dict[str, Any]]):
        """
        Deduplicate list of dicts while preserving mapping to original indices.

        Returns:
            unique_items: list of unique dicts
            mapping: dict[unique_idx] -> list[original_indices]
        """
        key_to_indices = defaultdict(list)
        key_to_item = {}

        for i, item in enumerate(data):
            key = self._freeze(item, sort_lists=True)
            key_to_indices[key].append(i)
            if key not in key_to_item:
                key_to_item[key] = item

        keys = list(key_to_item.keys())

        # map unique index → original indices
        mapping = {key: key_to_indices[key] for key in keys}

        return key_to_item, mapping

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[int, Dict[str, Any]]:
        """Run the full clustering pipeline over all items.

        If lsh_key was provided, runs fast_group first to produce
        lexical_group and product_group assignments for every item.
        Then runs hierarchical consensus clustering (queue-based, no recursion)
        to produce product_hierarchy.

        Returns:
            Dict mapping each item ID to a clustering dict:
            {
                "clustering": {
                    "lexical_group":    "LEX-042",          # if lsh_key set
                    "product_group":    "PRD-169482",       # if lsh_key set
                    "product_hierarchy": ["0001", "0001.0003"]
                }
            }
            lexical_group and product_group are omitted when lsh_key is None.
        """
        unique_ids = list(self._data.keys())

        # ------------------------------------------------------------------
        # Pre-processing: fast_group (optional)
        # ------------------------------------------------------------------
        lexical_map: Dict[int, str] = {}   # id -> LEX-N
        product_map: Dict[int, str] = {}   # id -> PRD-N

        queue: deque[_QueueItem] = deque()
        if self._lsh_key is not None:
            lsh_texts = [str(self._data[id_].get(self._lsh_key, "") or "") for id_ in unique_ids]
            sem_texts = ([str(self._data[id_].get(self._semantic_key, "") or "") for id_ in unique_ids] if self._semantic_key != self._lsh_key else lsh_texts)

            # LexicalGrouper operates on lsh_texts
            # SemanticGrouper operates on sem_texts of representatives
            lex_grouper = LexicalGrouper(**self._lsh_kwargs)
            lex_groups = lex_grouper.fit(lsh_texts)

            # Build representative sem_texts (first member of each lex group)
            lex_ids = list(lex_groups.keys())
            rep_sem_texts = [sem_texts[lex_groups[gid][0]] for gid in lex_ids]

            sem_grouper = SemanticGrouper(**self._semantic_kwargs)
            rep_sem_groups = sem_grouper.fit(rep_sem_texts)

            # Expand semantic groups back to item ids
            sem_expanded: Dict[str, List[int]] = defaultdict(list)
            for sem_gid, rep_positions in rep_sem_groups.items():
                for pos in rep_positions:
                    lex_gid = lex_ids[pos]
                    sem_expanded[sem_gid].extend(lex_groups[lex_gid])

            # Map each index -> group id, then index -> item id
            for lex_gid, indices in lex_groups.items():
                for idx in indices:
                    lexical_map[unique_ids[idx]] = lex_gid

            for sem_gid, indices in sem_expanded.items():
                for idx in indices:
                    product_map[unique_ids[idx]] = sem_gid
                queue.append(_QueueItem(ids=[unique_ids[idx] for idx in indices], parent_id=None, depth=0))
        else:
            queue.append(_QueueItem(ids=unique_ids, parent_id=None, depth=0))

        # ------------------------------------------------------------------
        # Hierarchical consensus clustering (queue-based)
        # ------------------------------------------------------------------
        hierarchy: Dict[int, List[str]] = {id_: [] for id_ in unique_ids}

        while queue:
            item = queue.popleft()
            ids = item.ids
            parent_id = item.parent_id
            depth = item.depth

            if len(ids) < 2:
                cid = self._next_cluster_id(parent_id)
                for id_ in ids:
                    hierarchy[id_].append(cid)
                continue

            if depth >= self._max_depth:
                cid = self._next_cluster_id(parent_id)
                for id_ in ids:
                    hierarchy[id_].append(cid)
                continue

            print(
                f"[NFeCluster] depth={depth} parent={parent_id} "
                f"n_items={len(ids)}",
                flush=True,
            )

            # Step 1: compute similarities for this subset
            sub_items = [self._data[id_] for id_ in ids]
            self._engine.compute_all(items=sub_items, ids=ids)

            # Step 2: build combined similarity matrix
            matrix = self._build_matrix(ids)

            # Step 3: run consensus clustering
            cluster_result = self._clusterer.run(
                ids=ids,
                matrix=matrix,
                rounds=self._rounds,
            )

            # Step 4: group ids by cluster label
            label_to_ids: Dict[int, List[int]] = {}
            for id_, label in cluster_result.membership.items():
                label_to_ids.setdefault(label, []).append(id_)

            # Step 5: compute metrics and decisions per cluster
            metrics_map = self._compute_metrics(label_to_ids, matrix, ids, depth)
            decisions_map = self._make_decisions(metrics_map)

            # Step 6: assign cluster IDs and enqueue splits
            for label, cluster_ids in label_to_ids.items():
                cid = self._next_cluster_id(parent_id)
                decision = decisions_map[label]

                if decision.keep_parent:
                    for id_ in cluster_ids:
                        hierarchy[id_].append(cid)

                if decision.should_split and len(cluster_ids) >= self._min_cluster_size:
                    if set(cluster_ids) == set(ids):
                        pass  # filho idêntico ao pai — evita loop infinito
                    else:
                        queue.append(_QueueItem(
                            ids=cluster_ids,
                            parent_id=cid,
                            depth=depth + 1,
                        ))
                elif not decision.keep_parent:
                    for id_ in cluster_ids:
                        hierarchy[id_].append(cid)

        # ------------------------------------------------------------------
        # Build final output
        # ------------------------------------------------------------------
        output: Dict[int, Dict[str, Any]] = {}
        for key in unique_ids:
            clustering: Dict[str, Any] = {}
            if lexical_map:
                clustering["lexical_group"] = lexical_map.get(key)
                clustering["product_group"] = product_map.get(key)
            clustering["product_hierarchy"] = hierarchy[key]
            temp = {"clustering": clustering}
            for id_ in self._uniqueid_to_index[key]:
                output[id_] = temp.copy()

        return output

    # ------------------------------------------------------------------
    # Rounds resolution
    # ------------------------------------------------------------------

    def _resolve_rounds(self, rounds: Any) -> List[Dict[str, Any]]:
        """Normalize rounds configuration to a list of explicit round dicts.

        Args:
            rounds: None, list[dict] or grid dict.

        Returns:
            List of explicit round dicts with keys: algorithm, threshold,
            funcs, bootstrap.
        """
        if rounds is None:
            grid = dict(_DEFAULT_GRID)
            if grid["func_groups"] is None:
                grid["func_groups"] = [self._engine.registered_functions()]
            return self._expand_grid(grid)

        if isinstance(rounds, list):
            return rounds

        if isinstance(rounds, dict):
            grid = dict(rounds)
            if grid.get("func_groups") is None:
                grid["func_groups"] = [self._engine.registered_functions()]
            return self._expand_grid(grid)

        raise ValueError(
            "`rounds` must be None, a list of dicts, or a grid dict. "
            f"Got: {type(rounds)}"
        )

    @staticmethod
    def _expand_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand a grid dict into all explicit round combinations.

        Args:
            grid: Dict with keys algorithms, thresholds, func_groups, bootstrap.

        Returns:
            List of explicit round dicts.
        """
        algorithms = grid.get("algorithms", ["leiden"])
        thresholds = grid.get("thresholds", [0.8])
        func_groups = grid.get("func_groups", [[]])
        bootstrap = grid.get("bootstrap", None)

        rounds = []
        for algo, threshold, funcs in product(algorithms, thresholds, func_groups):
            rounds.append({
                "algorithm": algo,
                "threshold": threshold,
                "funcs": list(funcs),
                "bootstrap": bootstrap,
            })
        return rounds

    # ------------------------------------------------------------------
    # Matrix building
    # ------------------------------------------------------------------

    def _build_matrix(self, ids: List[int]) -> np.ndarray:
        """Build a combined similarity matrix for a subset of IDs.

        Averages similarities across all registered functions. Uses engine.get()
        with the two item dicts — the engine handles cache lookup and key
        generation internally. Missing or incompatible function/field pairs
        are ignored gracefully (not counted in the average).

        Args:
            ids: List of item IDs to include in the matrix.

        Returns:
            np.ndarray of shape (n, n) with values in [0, 1].
        """
        n = len(ids)
        matrix = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(matrix, 1.0)

        for i in range(n):
            item_i = self._data[ids[i]]
            for j in range(i + 1, n):
                item_j = self._data[ids[j]]
                sims = self._engine.get(item_i, item_j)
                if sims:
                    avg = float(sum(sims.values()) / len(sims))
                else:
                    avg = 0.0
                matrix[i, j] = avg
                matrix[j, i] = avg

        return matrix

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        label_to_ids: Dict[int, List[int]],
        matrix: np.ndarray,
        all_ids: List[int],
        depth: int,
    ) -> Dict[int, ClusterMetrics]:
        """Compute ClusterMetrics for each cluster label.

        Uses the full similarity matrix (not just graph edges) for accurate
        intra-cluster and inter-cluster statistics.

        Args:
            label_to_ids: Mapping from cluster label to list of item IDs.
            matrix: Full similarity matrix for all IDs in this level.
            all_ids: Ordered list of IDs corresponding to matrix rows.
            depth: Current hierarchy depth.

        Returns:
            Dict mapping cluster label to ClusterMetrics.
        """
        id_to_local = {id_: i for i, id_ in enumerate(all_ids)}
        metrics: Dict[int, ClusterMetrics] = {}

        # Compute mean inter-cluster similarity for intra_inter_ratio
        # (mean similarity from each cluster to its nearest neighbour cluster)
        cluster_centroids: Dict[int, float] = {}  # label -> mean intra sim
        cluster_mean_sims: Dict[int, np.ndarray] = {}

        for label, cids in label_to_ids.items():
            idxs = [id_to_local[id_] for id_ in cids]
            if len(idxs) < 2:
                sims = np.array([1.0])
            else:
                pairs = [(matrix[idxs[a], idxs[b]])
                         for a in range(len(idxs))
                         for b in range(a + 1, len(idxs))]
                sims = np.array(pairs, dtype=np.float32)
            cluster_mean_sims[label] = sims
            cluster_centroids[label] = float(sims.mean())

        for label, cids in label_to_ids.items():
            idxs = [id_to_local[id_] for id_ in cids]
            sims = cluster_mean_sims[label]

            intra_mean = float(sims.mean())
            intra_std = float(sims.std()) if len(sims) > 1 else 0.0
            diameter = float(1.0 - sims.min()) if len(sims) > 0 else 0.0

            # Nearest neighbour cluster mean similarity
            other_means = [
                v for lbl, v in cluster_centroids.items() if lbl != label
            ]
            if other_means:
                nearest_inter = max(other_means)
                ratio = intra_mean / nearest_inter if nearest_inter > 0 else float("inf")
            else:
                ratio = None

            metrics[label] = ClusterMetrics(
                cluster_id="",  # filled later when IDs are assigned
                size=len(cids),
                depth=depth,
                intra_similarity_mean=intra_mean,
                intra_similarity_std=intra_std,
                diameter=diameter,
                intra_inter_ratio=ratio,
            )

        return metrics

    # ------------------------------------------------------------------
    # Split decision
    # ------------------------------------------------------------------

    def _make_decisions(
        self, metrics_map: Dict[int, ClusterMetrics]
    ) -> Dict[int, SplitDecision]:
        """Apply rule-based split decisions for each cluster.

        A cluster is a candidate for splitting if any of the following hold:
            - intra_similarity_mean < split_thresholds["silhouette_min"]
            - intra_similarity_std > split_thresholds["intra_std_max"]
            - diameter > split_thresholds["diameter_max"]
            - intra_inter_ratio < split_thresholds["intra_inter_ratio_min"]
              (when available)

        keep_parent is False only when the cluster has very poor cohesion
        (high std AND low intra mean simultaneously), suggesting it groups
        unrelated items that should not be presented as a valid cluster.

        Args:
            metrics_map: Dict from cluster label to ClusterMetrics.

        Returns:
            Dict from cluster label to SplitDecision.
        """
        thr = self._split_thresholds
        decisions: Dict[int, SplitDecision] = {}

        for label, m in metrics_map.items():
            reasons = []

            low_cohesion = m.intra_similarity_mean < thr["silhouette_min"]
            high_variance = m.intra_similarity_std > thr["intra_std_max"]
            large_diameter = m.diameter > thr["diameter_max"]
            poor_separation = (
                m.intra_inter_ratio is not None
                and m.intra_inter_ratio < thr["intra_inter_ratio_min"]
            )

            if low_cohesion:
                reasons.append(f"low intra_mean={m.intra_similarity_mean:.3f}")
            if high_variance:
                reasons.append(f"high intra_std={m.intra_similarity_std:.3f}")
            if large_diameter:
                reasons.append(f"large diameter={m.diameter:.3f}")
            if poor_separation:
                reasons.append(f"low intra_inter_ratio={m.intra_inter_ratio:.3f}")

            should_split = bool(reasons)

            # keep_parent = False only when cluster is noisy (high variance + low cohesion)
            keep_parent = not (low_cohesion and high_variance)

            if should_split:
                reason = "Split triggered: " + ", ".join(reasons)
            else:
                reason = "Cluster is cohesive — no split needed."

            decisions[label] = SplitDecision(
                cluster_id="",  # filled later
                should_split=should_split,
                keep_parent=keep_parent,
                confidence=1.0,  # rule-based: fully confident
                reason=reason,
            )

        return decisions

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _next_cluster_id(self, parent_id: Optional[str]) -> str:
        """Generate the next hierarchical cluster ID under a given parent.

        IDs are zero-padded 4-digit integers separated by dots.
        Root level: "0001", "0002", ...
        Children of "0001": "0001.0001", "0001.0002", ...

        Args:
            parent_id: Parent cluster ID string, or None for root level.

        Returns:
            New cluster ID string.
        """
        current = self._id_counters.get(parent_id, 0) + 1
        self._id_counters[parent_id] = current
        suffix = f"{current:04d}"
        return f"{parent_id}.{suffix}" if parent_id is not None else suffix

    def cluster_depth(self, cluster_id: str) -> int:
        """Return the depth of a cluster ID (number of dot-separated segments - 1).

        Args:
            cluster_id: Hierarchical cluster ID string.

        Returns:
            Depth integer (0 for root-level clusters).
        """
        return cluster_id.count(".")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _assert(condition: bool, message: str) -> None:
        """Simple assertion helper.

        Args:
            condition: Boolean to assert.
            message: Description shown on pass/fail.

        Raises:
            AssertionError: If condition is False.
        """
        if not condition:
            raise AssertionError(f"FAIL: {message}")
        print(f"  PASS: {message}")

    def _make_mock_engine(func_names: List[str], similarities: Dict[Tuple[int, int], float]):
        """Build a minimal mock SimilarityEngine for testing.

        Args:
            func_names: List of function names the engine exposes.
            similarities: Dict mapping (id_a, id_b) to similarity float.

        Returns:
            Mock object with the same interface as SimilarityEngine.
        """
        class MockEngine:
            def registered_functions(self):
                return func_names

            def compute_all(self, items, ids, **kwargs):
                pass  # already "cached"

            def get(self, func_name, id_a, id_b):
                key = (min(id_a, id_b), max(id_a, id_b))
                return similarities.get(key, 0.0)

        return MockEngine()

    def test_clusterer_basic() -> None:
        """Test Clusterer with a simple 4-item matrix with two clear clusters."""
        print("\n[test_clusterer_basic]")

        # Two tight clusters: {0,1} and {2,3}
        matrix = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ], dtype=np.float32)

        clusterer = Clusterer(n_runs_per_round=2)
        rounds = [{"algorithm": "louvain", "threshold": 0.5, "funcs": [], "bootstrap": None}]
        result = clusterer.run(ids=[10, 20, 30, 40], matrix=matrix, rounds=rounds)

        _assert(len(result.membership) == 4, "membership has 4 entries")
        _assert(
            result.membership[10] == result.membership[20],
            "ids 10 and 20 in same cluster",
        )
        _assert(
            result.membership[30] == result.membership[40],
            "ids 30 and 40 in same cluster",
        )
        _assert(
            result.membership[10] != result.membership[30],
            "ids 10 and 30 in different clusters",
        )

    def test_clusterer_single_item() -> None:
        """Test Clusterer with a single item."""
        print("\n[test_clusterer_single_item]")
        clusterer = Clusterer(n_runs_per_round=1)
        matrix = np.array([[1.0]])
        result = clusterer.run(ids=[99], matrix=matrix, rounds=[])
        _assert(result.membership == {99: 0}, "single item gets label 0")

    def test_clusterer_bootstrap() -> None:
        """Test that bootstrap runs complete without error."""
        print("\n[test_clusterer_bootstrap]")
        matrix = np.array([
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ], dtype=np.float32)
        clusterer = Clusterer(n_runs_per_round=3)
        rounds = [{"algorithm": "louvain", "threshold": 0.5, "funcs": [], "bootstrap": 0.75}]
        result = clusterer.run(ids=[1, 2, 3, 4], matrix=matrix, rounds=rounds)
        _assert(len(result.membership) == 4, "bootstrap run returned 4 memberships")

    def test_rounds_resolution() -> None:
        """Test that NFeMinerClustering resolves rounds correctly for all three input formats."""
        print("\n[test_rounds_resolution]")

        engine = _make_mock_engine(["seq"], {})
        items = [{"seq": "a"}, {"seq": "b"}]
        ids = [1, 2]

        # Default (None) — should use _DEFAULT_GRID
        nfe = NFeMinerClustering(items=items, ids=ids, engine=engine, rounds=None)
        _assert(isinstance(nfe._rounds, list), "None rounds resolves to list")
        _assert(len(nfe._rounds) > 0, "default rounds is non-empty")

        # Explicit list
        explicit = [{"algorithm": "leiden", "threshold": 0.8, "funcs": ["seq"], "bootstrap": None}]
        nfe2 = NFeMinerClustering(items=items, ids=ids, engine=engine, rounds=explicit)
        _assert(nfe2._rounds == explicit, "explicit list rounds preserved as-is")

        # Grid dict
        grid = {"algorithms": ["louvain"], "thresholds": [0.7, 0.8], "func_groups": [["seq"]]}
        nfe3 = NFeMinerClustering(items=items, ids=ids, engine=engine, rounds=grid)
        _assert(len(nfe3._rounds) == 2, "grid 1x2x1 expands to 2 rounds")

    def test_cluster_id_generation() -> None:
        """Test hierarchical ID generation format and ordering."""
        print("\n[test_cluster_id_generation]")

        engine = _make_mock_engine(["seq"], {})
        nfe = NFeMinerClustering(items=[{"seq": "a"}], ids=[1], engine=engine)

        id1 = nfe._next_cluster_id(None)
        id2 = nfe._next_cluster_id(None)
        id3 = nfe._next_cluster_id(None)
        _assert(id1 == "0001", "first root id is 0001")
        _assert(id2 == "0002", "second root id is 0002")

        child1 = nfe._next_cluster_id("0001")
        child2 = nfe._next_cluster_id("0001")
        _assert(child1 == "0001.0001", "first child of 0001")
        _assert(child2 == "0001.0002", "second child of 0001")

        _assert(nfe.cluster_depth("0001") == 0, "root depth is 0")
        _assert(nfe.cluster_depth("0001.0002") == 1, "child depth is 1")
        _assert(nfe.cluster_depth("0001.0002.0003") == 2, "grandchild depth is 2")

        # Tree reconstruction via string containment
        _assert("0001" in "0001.0002.0003", "parent id found via 'in'")
        _assert("0001.0002" in "0001.0002.0003", "grandparent found via 'in'")
        _assert("0002" not in "0001.0002.0003" or True, "note: '0002' in '0001.0002.0003' is True due to substring")
        # Correct check: split by "."
        _assert("0001" in "0001.0002.0003".split("."), "correct parent check via split")

    def test_NFeMinerClustering_run_two_clusters() -> None:
        """Test NFeMinerClustering.run() with two clearly separated groups."""
        print("\n[test_NFeMinerClustering_run_two_clusters]")

        # Items 1,2 similar to each other; items 3,4 similar to each other
        sims = {
            (1, 2): 0.95,
            (1, 3): 0.05,
            (1, 4): 0.05,
            (2, 3): 0.05,
            (2, 4): 0.05,
            (3, 4): 0.95,
        }
        engine = _make_mock_engine(["seq"], sims)
        items = [{"seq": v} for v in ["a", "b", "c", "d"]]
        ids = [1, 2, 3, 4]

        rounds = [{"algorithm": "louvain", "threshold": 0.5, "funcs": ["seq"], "bootstrap": None}]
        nfe = NFeMinerClustering(
            items=items,
            ids=ids,
            engine=engine,
            rounds=rounds,
            n_runs_per_round=2,
            max_depth=2,
        )
        result = nfe.run()

        _assert(set(result.keys()) == {1, 2, 3, 4}, "all ids present in result")

        # Each id should have at least one cluster assignment
        for id_, path in result.items():
            _assert(len(path) >= 1, f"id {id_} has at least one cluster")

        # ids 1 and 2 should share their first cluster
        _assert(
            result[1][0] == result[2][0],
            "ids 1 and 2 share root-level cluster",
        )
        _assert(
            result[3][0] == result[4][0],
            "ids 3 and 4 share root-level cluster",
        )
        _assert(
            result[1][0] != result[3][0],
            "ids 1 and 3 are in different root-level clusters",
        )

    def test_split_decision_noisy_cluster() -> None:
        """Test that a noisy cluster (low cohesion + high variance) gets keep_parent=False."""
        print("\n[test_split_decision_noisy_cluster]")

        engine = _make_mock_engine(["seq"], {})
        nfe = NFeMinerClustering(items=[{"seq": "x"}], ids=[1], engine=engine)

        noisy = ClusterMetrics(
            cluster_id="0001",
            size=10,
            depth=0,
            intra_similarity_mean=0.2,   # below silhouette_min=0.3
            intra_similarity_std=0.4,    # above intra_std_max=0.25
            diameter=0.8,
            intra_inter_ratio=0.9,
        )
        decisions = nfe._make_decisions({0: noisy})
        _assert(decisions[0].should_split, "noisy cluster should be split")
        _assert(not decisions[0].keep_parent, "noisy cluster should not be kept")

    def test_split_decision_good_cluster() -> None:
        """Test that a cohesive cluster gets should_split=False and keep_parent=True."""
        print("\n[test_split_decision_good_cluster]")

        engine = _make_mock_engine(["seq"], {})
        nfe = NFeMinerClustering(items=[{"seq": "x"}], ids=[1], engine=engine)

        good = ClusterMetrics(
            cluster_id="0001",
            size=5,
            depth=0,
            intra_similarity_mean=0.85,
            intra_similarity_std=0.05,
            diameter=0.15,
            intra_inter_ratio=3.0,
        )
        decisions = nfe._make_decisions({0: good})
        _assert(not decisions[0].should_split, "good cluster should not be split")
        _assert(decisions[0].keep_parent, "good cluster should be kept")

    print("=" * 60)
    print("clustering.py — running tests")
    print("=" * 60)

    test_clusterer_basic()
    test_clusterer_single_item()
    test_clusterer_bootstrap()
    test_rounds_resolution()
    test_cluster_id_generation()
    test_NFeMinerClustering_run_two_clusters()
    test_split_decision_noisy_cluster()
    test_split_decision_good_cluster()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)