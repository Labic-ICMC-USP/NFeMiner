from __future__ import annotations

"""
Edge generator with LMDB-backed persistent cache and igraph output.

Design:
  - Input: `items` (list[str]), `ids` (list[int]) with possible repeated texts.
  - Constructor performs normalization (lower().strip()), contraction (unique texts),
    inspects LMDB cache for similarities between unique representatives, computes
    missing similarities in parallel with ProcessPoolExecutor and shows tqdm progress.
  - LMDB key: big-endian pair of uint64 (min_id, max_id).
  - LMDB value: binary float16 or float32 (configurable).
  - generate_graph(threshold) reads LMDB, expands unique-pair similarities to all
    original ids, adds full-cliques for identical texts (sim = 1.0), and returns:
        args: dict of hyperparameters used
        graph: igraph.Graph with vertex attribute 'id', edge attributes 'weight' and 'edge_type'
"""

import os, struct, uuid, lmdb, numpy as np,  igraph as ig
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Any, Dict, Iterable, List, Tuple
from tqdm import tqdm
from difflib import SequenceMatcher

# -------------------------
# Top-level compare registry
# -------------------------
# Worker processes will look up comparison function by name from this registry.
# Add other compare functions here if you implement other EdgeGenerator subclasses.
def sequence_compare(a: str, b: str) -> float:
    """Compare two strings using difflib.SequenceMatcher.ratio()."""
    return SequenceMatcher(None, a, b).ratio()


_COMPARE_REGISTRY = {
    "sequence": sequence_compare,
    # future: "rapidfuzz": rapidfuzz_compare, etc.
}


# -------------------------
# Top-level worker
# -------------------------
def _worker_task(task: Tuple[int, str, int, str, str]) -> Tuple[int, int, float]:
    """
    Top-level worker executed in child processes.

    Args:
        task: (i, text_i, j, text_j, compare_name)

    Returns:
        (i, j, similarity)
    """
    i, a, j, b, compare_name = task
    cmp_fn = _COMPARE_REGISTRY[compare_name]
    sim = float(cmp_fn(a, b))
    return i, j, sim


# -------------------------
# EdgeGenerator base class
# -------------------------
class EdgeGenerator(ABC):
    """
    Base class that computes pairwise similarities between unique normalized texts,
    caches them in LMDB and builds igraph graphs filtered by a threshold.

    Subclasses must implement `_compute_similarity(a, b)` or choose a `compare_name`
    that points into the `_COMPARE_REGISTRY` above.

    Example:
        gen = StringMatchEdgeGenerator(items, ids, cache_dir)
        args, graph = gen.generate_graph(threshold=0.85)

    Args:
        items: list[str] containing text values (may contain duplicates).
        ids: list[int] same length as items, original integer PRIMARY KEYs (int64).
        cache_path: directory where LMDB files are stored.
        similarity_dtype: 'float16' or 'float32' (default 'float16').
        max_workers: number of parallel workers; default = os.cpu_count().
        batch_size: internal batch size for writing to LMDB & memory (default 50k).
    """

    def __init__(
        self,
        items: List[str],
        ids: List[int],
        cache_path: str,
        similarity_dtype: str = "float16",
        max_workers: int | None = None,
        batch_size: int = 50_000,
    ):
        # Validate inputs
        if len(items) != len(ids):
            raise ValueError("`items` and `ids` must have the same length.")
        self.items: List[str] = list(items)
        self.ids: List[int] = [int(x) for x in ids]
        self.n_total = len(self.items)
        self.cache_path = str(cache_path)
        os.makedirs(self.cache_path, exist_ok=True)

        if similarity_dtype not in ("float16", "float32"):
            raise ValueError("similarity_dtype must be 'float16' or 'float32'")
        self.similarity_dtype = similarity_dtype
        self.max_workers = max_workers or (os.cpu_count() or 1)
        self.batch_size = int(batch_size)

        # Edge type label will be provided by subclass (default: class name)
        self.edge_type = getattr(self, "EDGE_TYPE", self.__class__.__name__)

        # Step 1: normalize and contract duplicates
        # Normalization used for contraction: lower().strip() (as requested)
        self._normalized = [s.lower().strip() for s in self.items]
        # Map normalized_text -> list of original ids (original id values)
        self.normal_to_ids: Dict[str, List[int]] = {}
        for idx, txt in enumerate(self._normalized):
            original_id = self.ids[idx]
            self.normal_to_ids.setdefault(txt, []).append(original_id)

        # Build list of unique normalized texts and representative IDs (first id in group)
        self.unique_texts: List[str] = []
        self.rep_ids: List[int] = []  # representative id (int64) for each unique text
        self.rep_index_by_text: Dict[str, int] = {}

        for txt, id_list in self.normal_to_ids.items():
            self.rep_index_by_text[txt] = len(self.unique_texts)
            self.unique_texts.append(txt)
            self.rep_ids.append(int(id_list[0]))  # pick first as representative

        self.n_unique = len(self.unique_texts)

        # Build mapping original id -> index in self.ids (for graph vertex indexing)
        self.id_to_index: Dict[int, int] = {int(idv): i for i, idv in enumerate(self.ids)}

        # LMDB environment path (one DB per subclass class name)
        db_filename = f"{self.__class__.__name__}.lmdb"
        self.lmdb_path = os.path.join(self.cache_path, db_filename)
        self.env = lmdb.open(
            self.lmdb_path,
            map_size=1024 ** 4,
            subdir=False,
            readonly=False,
            create=True,
            metasync=False,
            sync=False,
            map_async=True,
        )

        # Show informational message
        total_pairs = self.n_unique * (self.n_unique - 1) // 2
        print(f"[EdgeGenerator] {self.n_total} input items -> {self.n_unique} unique texts.")
        print(f"[EdgeGenerator] Up to {total_pairs:,} unique pairs to cache (between unique texts).", flush=True)

        # Prepare compare_name used by worker; subclasses can override compare_name
        # Default: 'sequence' -> uses SequenceMatcher
        self.compare_name = getattr(self, "COMPARE_NAME", "sequence")
        if self.compare_name not in _COMPARE_REGISTRY:
            raise ValueError(f"compare_name '{self.compare_name}' not found in registry")

        # Now inspect LMDB and compute missing similarities (if any).
        self._compute_missing_similarities_with_progress()

    # -------------------------
    # Packing helpers
    # -------------------------
    @staticmethod
    def _pack_key(u: int, v: int) -> bytes:
        """Pack two uint64 integers into a big-endian 16-byte key (min,max)."""
        u = int(u)
        v = int(v)
        if u <= v:
            return struct.pack(">QQ", u, v)
        return struct.pack(">QQ", v, u)

    def _pack_value(self, sim: float) -> bytes:
        """Pack similarity float into bytes according to self.similarity_dtype."""
        if self.similarity_dtype == "float16":
            arr = np.array([sim], dtype=np.float16)
            return arr.tobytes()
        return np.array([sim], dtype=np.float32).tobytes()

    def _unpack_value(self, data: bytes) -> float:
        """Unpack similarity bytes according to dtype and return Python float."""
        if self.similarity_dtype == "float16":
            return float(np.frombuffer(data, dtype=np.float16)[0])
        return float(np.frombuffer(data, dtype=np.float32)[0])

    # -------------------------
    # Subclasses should implement or set compare_name
    # -------------------------
    @abstractmethod
    def _compute_similarity(self, a: str, b: str) -> float:
        """
        Compute similarity between two strings.

        Subclasses should implement this method OR set `COMPARE_NAME` to a key present
        in the module-level `_COMPARE_REGISTRY`. The default `StringMatchEdgeGenerator`
        implements this using difflib.SequenceMatcher.ratio().
        """
        raise NotImplementedError

    # -------------------------
    # Internal: create tasks generator for missing pairs
    # -------------------------
    def _iter_unique_pairs(self) -> Iterable[Tuple[int, str, int, str]]:
        """Yield (i, text_i, j, text_j) for i < j over unique texts."""
        for i in range(self.n_unique):
            ti = self.unique_texts[i]
            for j in range(i + 1, self.n_unique):
                yield (i, ti, j, self.unique_texts[j])

    # -------------------------
    # Compute missing similarities (constructor action)
    # -------------------------
    def _compute_missing_similarities_with_progress(self) -> None:
        """
        Inspect LMDB to count how many unique pairs are already cached, then
        compute the missing similarities in parallel while showing a tqdm progress bar.
        """
        total_pairs = self.n_unique * (self.n_unique - 1) // 2
        if total_pairs <= 0:
            print("[EdgeGenerator] No unique pairs to compute (0 or 1 unique items).")
            return

        # First pass: count missing pairs (fast read-only txn)
        missing = 0
        with self.env.begin(write=False) as txn:
            for (i, _, j, _) in self._iter_unique_pairs():
                key = self._pack_key(self.rep_ids[i], self.rep_ids[j])
                if txn.get(key) is None:
                    missing += 1

        print(f"[EdgeGenerator] {total_pairs:,} total unique pairs; {missing:,} missing and will be computed now.", flush=True)

        if missing == 0:
            return

        # Prepare tasks generator for missing pairs (we re-iterate)
        def missing_tasks_gen():
            with self.env.begin(write=False) as txn_read:
                for (i, ti, j, tj) in self._iter_unique_pairs():
                    key = self._pack_key(self.rep_ids[i], self.rep_ids[j])
                    if txn_read.get(key) is None:
                        # include compare_name so workers can pick the function from registry
                        yield (i, ti, j, tj, self.compare_name)

        # Process tasks in parallel and write in batches to LMDB
        tasks_iter = missing_tasks_gen()

        # We'll use ProcessPoolExecutor and iterate results to update progress
        processed = 0
        batch_writes: List[Tuple[int, int, float]] = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            # ex.map returns results in task order (deterministic). We set chunksize modestly.
            for (i, j, sim) in tqdm(ex.map(_worker_task, tasks_iter, chunksize=512), total=missing, desc="Computing similarities"):
                batch_writes.append((i, j, sim))
                processed += 1

                if len(batch_writes) >= self.batch_size:
                    # flush batch
                    with self.env.begin(write=True) as wtxn:
                        for (ii, jj, s) in batch_writes:
                            key = self._pack_key(self.rep_ids[ii], self.rep_ids[jj])
                            wtxn.put(key, self._pack_value(s))
                    batch_writes = []

            # flush remaining
            if batch_writes:
                with self.env.begin(write=True) as wtxn:
                    for (ii, jj, s) in batch_writes:
                        key = self._pack_key(self.rep_ids[ii], self.rep_ids[jj])
                        wtxn.put(key, self._pack_value(s))
                batch_writes = []

        print(f"[EdgeGenerator] Computed and wrote {processed:,} similarities to LMDB: {self.lmdb_path}", flush=True)

    def generate_compact_graph(self, threshold: float = 0.0) -> ig.Graph:
        """
        Generate a compact igraph.Graph containing only unique texts (representatives).

        The method reads pairwise similarities from the LMDB cache (stored between
        representative ids), applies the provided threshold and builds an igraph
        that has `n_unique` vertices. Each vertex receives:
        - "rep_id": representative original id chosen for the unique text
        - "text": the normalized unique text

        Edges are created only between unique representatives whose cached
        similarity >= threshold and the edge attribute "weight" receives the similarity.

        Args:
            threshold: float in [0,1]. Only edges with similarity >= threshold will
                    be created in the compact graph.

        Returns:
            igraph.Graph: compact graph with n_unique vertices and weighted edges.
        """
        # Prepare result graph with n_unique vertices
        n = self.n_unique
        g = ig.Graph()
        g.add_vertices(n)

        # attach representative id and normalized text as vertex attributes
        # rep_id corresponds to self.rep_ids list; text corresponds to self.unique_texts
        g.vs["rep_id"] = self.rep_ids
        g.vs["text"] = self.unique_texts

        # build a mapping rep_id -> rep_index for fast lookup
        rep_id_to_index = {int(rid): idx for idx, rid in enumerate(self.rep_ids)}

        edges = []
        weights = []

        # read LMDB entries (keys are packed >QQ)
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                try:
                    u_rep, v_rep = struct.unpack(">QQ", key)
                except Exception:
                    # skip malformed key
                    continue

                # Get representative indexes; if not found, skip (defensive)
                i_rep = rep_id_to_index.get(int(u_rep))
                j_rep = rep_id_to_index.get(int(v_rep))
                if i_rep is None or j_rep is None:
                    continue

                sim = self._unpack_value(val)
                if sim >= threshold:
                    edges.append((i_rep, j_rep))
                    weights.append(sim)

        # add edges (if any) and set weight attribute
        if edges:
            g.add_edges(edges)
            g.es["weight"] = weights

        return g

    def expand_clusters(self, compact_labels, params):
        """
        Expand compact graph cluster labels back to all original IDs.

        Args:
            compact_labels (List[int]):
                Cluster label for each representative node (length = n_unique).

            params (Dict[str, Any]):
                Additional metadata (algorithm name, threshold, hyperparameters).
                A 'cluster' field will be added automatically for each id.

        Returns:
            List[Tuple[int, Dict[str, Any]]]:
                One entry per original ID: (id_original, params_with_cluster)
        """

        # 1. Map compact cluster labels → unique UUID per cluster
        cluster_ids = {}
        for lbl in set(compact_labels):
            cluster_ids[lbl] = str(uuid.uuid4())

        # 2. Output list
        out = []
        
        args_base = {"edge_type": self.edge_type, "generator": self.__class__.__name__}

        # 3. For each unique normalized text (representative)
        for idx_unique, rep_id in enumerate(self.rep_ids):
            compact_label = compact_labels[idx_unique]
            cluster_uuid = cluster_ids[compact_label]

            # All original IDs that belong to this representative text
            norm_text = self.unique_texts[idx_unique]
            original_ids = self.normal_to_ids[norm_text]

            for oid in original_ids:
                # clone params dict to avoid mutating external references
                p = dict(args_base) | dict(params)
                p["cluster"] = cluster_uuid
                out.append((oid, p))

        return out


# -------------------------
# Concrete Generator: SequenceMatcher
# -------------------------
class StringMatchEdgeGenerator(EdgeGenerator):
    """
    Edge generator using difflib.SequenceMatcher as similarity measure.

    This class does not need to override the heavy-lifting caching logic;
    it only provides the comparison name which the worker uses.
    """

    EDGE_TYPE = "string_match"
    COMPARE_NAME = "sequence"

    def _compute_similarity(self, a: str, b: str) -> float:  # pragma: no cover - not used directly by workers
        """Return SequenceMatcher ratio (kept for API completeness)."""
        return SequenceMatcher(None, a, b).ratio()