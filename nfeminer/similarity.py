"""
Module for computing pairwise similarities between items, with persistent caching
via KVStore and parallel computation.

Design:
    Four base classes form a hierarchy:

    SimilarityFunction
        Base for single-key, pair-by-pair functions. Receives one value extracted
        from the item dict via a single key. Parallelized with ProcessPoolExecutor.

    BatchSimilarityFunction(SimilarityFunction)
        Base for single-key functions where batch processing is significantly
        faster (e.g. BERT). All missing pairs are sent to compute_batch in one
        call instead of being distributed across workers.

    MultiKeySimilarityFunction(SimilarityFunction)
        Base for functions that need more than one field from the item dict.
        extract() returns a dict {key: value} instead of a single value.
        The compute() implementation decides how to combine multiple fields.

    MultiKeyBatchSimilarityFunction(BatchSimilarityFunction, MultiKeySimilarityFunction)
        Combines multi-key extraction with batch processing. Detected by the
        engine via isinstance(fn, BatchSimilarityFunction) — no engine changes needed.

    SimilarityEngine
        Orchestrates computation, caching (KVStore) and parallelization.
        Items are always passed as dicts keyed by field name. Each function
        calls self.extract(item) internally to get its value(s).

Usage:
    engine = SimilarityEngine(
        funcs=[
            SequenceMatchSimilarity('produto_base'),
            SequenceMatchSimilarity('produto_detalhado'),
            BERTSimilarity('produto_base'),
            NCMSimilarity('ncm'),
            CategorySimilarity('categoria'),
            TagSimilarity('tags'),
        ],
        cache=KVStore(...),
        max_workers=4,
    )
    engine.compute_all(items=[{"produto_base": "Agua Mineral", ...}], ids=[1, 2])
    sims = engine.get(
        {"produto_base": "Agua Mineral", "ncm": "22011000"},
        {"produto_base": "Agua com Gas"},
    )
    # sims = {("sequence_match__produto_base", "produto_base"): 0.61, ...}
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class SimilarityFunction(ABC):
    """Base class for single-key, pair-by-pair similarity functions.

    Use this class when:
        - The function needs exactly one field from the item dict.
        - The cost per pair is constant and independent of other pairs.

    Each pair is computed individually and distributed across ProcessPoolExecutor
    workers. The engine extracts the relevant field by calling self.extract(item)
    before passing values to compute().

    Subclasses must define:
        - ``base_name`` (str): algorithm identifier, e.g. "sequence_match".
        - ``compute(a, b) -> float``: similarity in [0.0, 1.0].

    The ``name`` attribute is auto-generated as ``f"{base_name}__{key}"`` and
    is used as the cache key prefix. This guarantees that two instances of the
    same algorithm operating on different fields never collide in the cache.

    Args:
        key: Field name to extract from item dicts.

    Example:
        funcs = [
            SequenceMatchSimilarity('produto_base'),
            SequenceMatchSimilarity('produto_detalhado'),
        ]
        # names: "sequence_match__produto_base", "sequence_match__produto_detalhado"
    """

    base_name: str

    def __init__(self, key: str) -> None:
        """Initialize with the dict key to extract.

        Args:
            key: Field name in the item dict this function operates on.
        """
        self.key = key
        self.name = self.base_name

    def extract(self, item: Any) -> Any:
        """Extract the relevant value from an item dict.

        Args:
            item: Item dict or raw value. If dict, returns item.get(self.key).
                  If not a dict, returns item directly (for backward compat).

        Returns:
            Extracted value for this function's key.
        """
        if isinstance(item, dict):
            return item.get(self.key)
        return item

    @abstractmethod
    def compute(self, a: Any, b: Any) -> float:
        """Compute similarity between two extracted values.

        Args:
            a: First extracted value.
            b: Second extracted value.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        raise NotImplementedError

class BatchSimilarityFunction(SimilarityFunction):
    """Base class for single-key functions where batch processing is significantly faster.

    Use this class when computing N pairs together is much cheaper than computing
    them one by one. The canonical example is BERT: encoding all texts in a single
    forward pass is orders of magnitude faster than N individual passes.

    Suitable for:
        - Sentence/word embeddings (BERT, SBERT, FastText)
        - TF-IDF vectorization (fit once, transform N texts together)
        - Any model with a GPU/vectorized batch inference path

    The SimilarityEngine detects BatchSimilarityFunction via isinstance() and
    routes all missing pairs to compute_batch in a single call.

    Subclasses must implement:
        - ``base_name`` (str)
        - ``compute(a, b) -> float``: single-pair fallback.
        - ``compute_batch(pairs) -> list[float]``: vectorized batch.

    Args:
        key: Field name to extract from item dicts.
    """

    @abstractmethod
    def compute_batch(self, pairs: List[Tuple[Any, Any]]) -> List[float]:
        """Compute similarity for a batch of pairs.

        Args:
            pairs: List of (a, b) tuples of already-extracted values.

        Returns:
            List of similarity scores in [0.0, 1.0], same order as input.
        """
        raise NotImplementedError

class MultiKeySimilarityFunction(SimilarityFunction):
    """Base class for pair-by-pair functions that need multiple fields.

    Use this class when the similarity computation requires more than one field
    from the item dict (e.g. combining unit of measure and numeric value).

    extract() returns a dict {key: value} for all requested keys. The compute()
    implementation receives these dicts and decides how to combine them.

    The ``name`` is auto-generated as ``f"{base_name}__{','.join(keys)}"``.

    Args:
        keys: List of field names to extract from item dicts.

    Example:
        class PriceUnitSimilarity(MultiKeySimilarityFunction):
            base_name = "price_unit"

            def compute(self, a: dict, b: dict) -> float:
                if a["unit"] != b["unit"]:
                    return 0.0
                return 1.0 - abs(a["value"] - b["value"]) / max_range
    """

    def __init__(self, keys: List[str]) -> None:
        """Initialize with multiple dict keys.

        Args:
            keys: Field names in the item dict this function operates on.

        Raises:
            ValueError: If keys is empty.
        """
        if not keys:
            raise ValueError("keys must not be empty.")
        self.keys = list(keys)
        self.key = keys[0]  # kept for interface compatibility
        self.name = f"{self.base_name}__{','.join(keys)}"

    def extract(self, item: Any) -> Dict[str, Any]:
        """Extract all relevant fields from an item dict.

        Args:
            item: Item dict. Non-dict items return an empty dict.

        Returns:
            Dict mapping each key to its value in the item dict.
        """
        if isinstance(item, dict):
            return {k: item.get(k) for k in self.keys}
        return {}

    @abstractmethod
    def compute(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Compute similarity between two multi-field dicts.

        Args:
            a: Dict of extracted fields for item A.
            b: Dict of extracted fields for item B.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        raise NotImplementedError

class MultiKeyBatchSimilarityFunction(BatchSimilarityFunction, MultiKeySimilarityFunction):
    """Base class combining multi-key extraction with batch processing.

    Use this class when:
        - The function needs more than one field from the item dict, AND
        - Batch processing is significantly more efficient than pair-by-pair.

    Because this inherits from BatchSimilarityFunction, the engine detects it
    via isinstance(fn, BatchSimilarityFunction) and routes all pairs to
    compute_batch — no engine changes required.

    MRO resolves extract() from MultiKeySimilarityFunction since
    BatchSimilarityFunction does not override it.

    Args:
        keys: List of field names to extract from item dicts.

    Subclasses must implement:
        - ``base_name`` (str)
        - ``compute(a, b) -> float``: single-pair fallback.
        - ``compute_batch(pairs) -> list[float]``: vectorized batch.
    """

    def __init__(self, keys: List[str]) -> None:
        """Initialize with multiple dict keys.

        Args:
            keys: Field names in the item dict.
        """
        MultiKeySimilarityFunction.__init__(self, keys)

# ---------------------------------------------------------------------------
# Concrete implementations — single key
# ---------------------------------------------------------------------------

class SequenceMatchSimilarity(SimilarityFunction):
    """String similarity using difflib.SequenceMatcher.

    Computes the ratio of matching characters between two strings after
    lowercasing and stripping whitespace. Good as a lightweight baseline
    for product description matching.

    Args:
        key: Field name in the item dict containing the string to compare.

    Example:
        SequenceMatchSimilarity('produto_base')
        SequenceMatchSimilarity('produto_detalhado')
    """

    base_name = "sequence_match"

    def compute(self, a: Any, b: Any) -> float:
        """Compute SequenceMatcher ratio between two strings.

        Args:
            a: First string (or None).
            b: Second string (or None).

        Returns:
            Similarity ratio in [0.0, 1.0]. Returns 0.0 if either value is None.
        """
        if a is None or b is None:
            return 0.0
        return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()

class NCMSimilarity(SimilarityFunction):
    """Similarity based on NCM fiscal code prefix matching.

    NCM codes are hierarchical: the more digits two codes share from the
    left, the more related the products are. Missing NCM values yield 0.0.
    Codes are zero-padded to 8 digits before comparison.

    Args:
        key: Field name in the item dict containing the NCM code string.
    """

    base_name = "ncm"

    def compute(self, a: Any, b: Any) -> float:
        """Compute similarity based on shared NCM prefix length.

        Args:
            a: NCM code string (e.g. "01012100") or None.
            b: NCM code string or None.

        Returns:
            Similarity in [0.0, 1.0]. Returns 0.0 if either value is None.
        """
        if not a or not b:
            return 0.0
        a = str(a).strip().zfill(8)
        b = str(b).strip().zfill(8)
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 0.0
        prefix = 0
        for x, y in zip(a, b):
            if x != y:
                break
            prefix += 1
        return prefix / max_len

class CategorySimilarity(SimilarityFunction):
    """Similarity between two hierarchical category lists using Jaccard overlap.

    Args:
        key: Field name in the item dict containing the category list.
    """

    base_name = "category"

    def compute(self, a: Any, b: Any) -> float:
        """Compute Jaccard similarity between two category lists.

        Args:
            a: List of category strings for item A (or None).
            b: List of category strings for item B (or None).

        Returns:
            Jaccard similarity in [0.0, 1.0]. Returns 0.0 if both are empty.
        """
        set_a = set(str(x).lower().strip() for x in (a or []))
        set_b = set(str(x).lower().strip() for x in (b or []))
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

class TagSimilarity(SimilarityFunction):
    """Similarity between two flat tag lists using Jaccard overlap.

    Args:
        key: Field name in the item dict containing the tag list.
    """

    base_name = "tags"

    def compute(self, a: Any, b: Any) -> float:
        """Compute Jaccard similarity between two tag lists.

        Args:
            a: List of tag strings for item A (or None).
            b: List of tag strings for item B (or None).

        Returns:
            Jaccard similarity in [0.0, 1.0]. Returns 0.0 if both are empty.
        """
        set_a = set(str(x).lower().strip() for x in (a or []))
        set_b = set(str(x).lower().strip() for x in (b or []))
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

class BERTSimilarity(BatchSimilarityFunction):
    """Semantic similarity using Sentence-BERT embeddings.

    Encodes all unique texts in a single batch forward pass, then computes
    cosine similarity between embedding pairs. Significantly faster than
    calling the model once per pair.

    Args:
        key: Field name in the item dict containing the text string.
        model_name: SentenceTransformer model identifier.
    """

    base_name = "bert"

    def __init__(self, key: str, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize and load the SentenceTransformer model.

        Args:
            key: Field name in the item dict containing the text to encode.
            model_name: HuggingFace / SentenceTransformer model name.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        super().__init__(key)
        self.name = f'{self.base_name}__{model_name}'
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for BERTSimilarity. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def compute(self, a: Any, b: Any) -> float:
        """Compute BERT cosine similarity for a single pair.

        Args:
            a: First text string (or None).
            b: Second text string (or None).

        Returns:
            Cosine similarity in [0.0, 1.0]. Returns 0.0 if either is None.
        """
        if a is None or b is None:
            return 0.0
        return self.compute_batch([(a, b)])[0]

    def compute_batch(self, pairs: List[Tuple[Any, Any]]) -> List[float]:
        """Compute BERT cosine similarity for a batch of pairs.

        All unique texts are encoded in a single forward pass, then cosine
        similarity is computed via vectorized dot products.

        Args:
            pairs: List of (text_a, text_b) pairs of already-extracted strings.

        Returns:
            List of cosine similarities in [0.0, 1.0], same order as input.
        """
        import torch
        import torch.nn.functional as F

        unique_texts = list({str(t) for pair in pairs for t in pair if t is not None})
        if not unique_texts:
            return [0.0] * len(pairs)

        text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        embeddings = self._model.encode(
            unique_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)

        results = []
        for a, b in pairs:
            if a is None or b is None:
                results.append(0.0)
                continue
            i = text_to_idx[str(a)]
            j = text_to_idx[str(b)]
            sim = float(torch.dot(embeddings[i], embeddings[j]).clamp(0.0, 1.0))
            results.append(sim)
        return results

# ---------------------------------------------------------------------------
# Concrete implementations — multi key
# ---------------------------------------------------------------------------

class NumericRangeSimilarity(MultiKeySimilarityFunction):
    """Similarity between two numeric values with an optional group constraint.

    Expects item dicts with a numeric value field and an optional group field
    (e.g. unit of measure). Pairs from different groups get 0.0.

    Args:
        value_key: Field name containing the numeric value.
        group_key: Field name containing the group label. None to disable.
        min_value: Lower bound of the numeric band.
        max_value: Upper bound of the numeric band.

    Example:
        NumericRangeSimilarity(
            value_key="valor_unitario",
            group_key="unidade_comercializacao",
            min_value=0.0,
            max_value=1000.0,
        )
    """

    base_name = "numeric_range"

    def __init__(
        self,
        value_key: str,
        group_key: Optional[str] = None,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> None:
        """Initialize with field names and numeric band.

        Args:
            value_key: Field name for the numeric value.
            group_key: Field name for the group label, or None.
            min_value: Lower bound.
            max_value: Upper bound.

        Raises:
            ValueError: If min_value >= max_value.
        """
        if min_value >= max_value:
            raise ValueError("min_value must be strictly less than max_value.")
        keys = [value_key] + ([group_key] if group_key else [])
        super().__init__(keys)
        self.name = f'{self.base_name}__{float(min_value)}__{float(max_value)}'
        self.value_key = value_key
        self.group_key = group_key
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def compute(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Compute numeric band similarity between two multi-field dicts.

        Args:
            a: Dict with value_key (and optionally group_key) for item A.
            b: Dict with value_key (and optionally group_key) for item B.

        Returns:
            Similarity in [0.0, 1.0].
        """
        val_a = a.get(self.value_key)
        val_b = b.get(self.value_key)
        if val_a is None or val_b is None:
            return 0.0
        try:
            val_a, val_b = float(val_a), float(val_b)
        except (TypeError, ValueError):
            return 0.0
        if self.group_key:
            grp_a, grp_b = a.get(self.group_key), b.get(self.group_key)
            if grp_a is not None and grp_b is not None and grp_a != grp_b:
                return 0.0
        band = self.max_value - self.min_value
        if not (self.min_value <= val_a <= self.max_value):
            return 0.0
        if not (self.min_value <= val_b <= self.max_value):
            return 0.0
        return 1.0 - abs(val_a - val_b) / band

# ---------------------------------------------------------------------------
# Worker (top-level for pickling with ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_compute(task: Tuple[str, Any, Any]) -> float:
    """Top-level worker for parallel pair-by-pair computation.

    Args:
        task: Tuple of (func_name, val_a, val_b).
              val_a and val_b are already extracted by fn.extract().

    Returns:
        Similarity float.

    Note:
        Uses the module-level _FUNCTION_REGISTRY populated before spawning workers.
    """
    func_name, val_a, val_b = task
    func = _FUNCTION_REGISTRY[func_name]
    return float(func.compute(val_a, val_b))

# Registry populated by SimilarityEngine before spawning workers
_FUNCTION_REGISTRY: Dict[str, SimilarityFunction] = {}

# ---------------------------------------------------------------------------
# SimilarityEngine
# ---------------------------------------------------------------------------

class SimilarityEngine:
    """Orchestrates similarity computation with persistent caching and parallelization.

    For each registered SimilarityFunction, computes all pairwise similarities
    between a set of items and stores results in a KVStore cache. Already cached
    pairs are skipped, making the operation idempotent and resumable.

    Cache key design:
        Keys are based on the extracted *values*, not on item IDs. This means
        two items with the same extracted value (e.g. same "produto_base" text)
        share a cache entry regardless of which dataset or ID they came from.
        Keys are opaque MD5 hashes — use get() and compute_all() to interact
        with the cache; never construct keys manually.

    Routing strategy:
        - isinstance(fn, BatchSimilarityFunction): all missing pairs sent to
          compute_batch in one call (includes MultiKeyBatchSimilarityFunction).
        - Otherwise: missing pairs distributed across ProcessPoolExecutor workers.

    Args:
        funcs: List of SimilarityFunction instances (any subclass).
        cache: KVStore instance with KeyMode.SINGLE_KEY and ValueMode.NUMERIC.
        max_workers: Workers for ProcessPoolExecutor. Defaults to cpu_count - 1.
        batch_size: Number of results to accumulate before flushing to cache.

    Example:
        engine = SimilarityEngine(
            funcs=[
                SequenceMatchSimilarity('produto_base'),
                BERTSimilarity('produto_base'),
                NCMSimilarity('ncm'),
            ],
            cache=kvstore,
        )
        engine.compute_all(
            items=[{"produto_base": "Agua Mineral", "ncm": "22011000"}],
            ids=[1],
        )
        sims = engine.get(
            {"produto_base": "Agua Mineral"},
            {"produto_base": "Agua com Gas"},
        )
        # {("bert__all-MiniLM-L6-v2", "produto_base"): 0.72,
        #  ("sequence_match", "produto_base"): 0.61}
    """

    def __init__(
        self,
        funcs: List[SimilarityFunction],
        cache: Any,
        max_workers: Optional[int] = None,
        batch_size: int = 50_000,
    ) -> None:
        """Initialize the SimilarityEngine.

        Args:
            funcs: Similarity functions to register.
            cache: KVStore instance for persistent caching.
            max_workers: Workers for ProcessPoolExecutor. Defaults to cpu_count - 1.
            batch_size: Cache flush interval (number of results).

        Raises:
            ValueError: If two functions share the same name.
        """
        self._funcs: Dict[str, SimilarityFunction] = {}
        for fn in funcs:
            if fn.name in self._funcs:
                raise ValueError(
                    f"Duplicate similarity function name: '{fn.name}'. "
                    "Two instances of the same algorithm must use different keys."
                )
            self._funcs[fn.name] = fn

        self._cache = cache
        self._max_workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(
        self,
        items: List[Any],
        func_names: Optional[List[str]] = None,
        use_sampling: bool = False,
        sample_size: Optional[int] = None,
    ) -> None:
        """Compute and cache all pairwise similarities for the given items.

        Already cached pairs are skipped. The operation is idempotent.
        Duplicate items (same extracted value) are contracted per function —
        their similarity is computed once and shared across any dataset.

        Execution strategy:
            - Parallel functions (SimilarityFunction but not BatchSimilarityFunction)
              share a single ProcessPoolExecutor. For each function, pairs are
              streamed from a generator and submitted to the executor as soon as
              a cache miss is found — workers run while the main thread keeps
              scanning for more missing pairs.
            - Batch functions (BatchSimilarityFunction) run after all parallel
              functions finish, processed in chunks to avoid materializing all
              pairs in memory at once.

        Args:
            items: List of item dicts keyed by field name.
            func_names: Names of functions to run. Defaults to all registered.
            use_sampling: If True, only a random sample of pairs is computed.
            sample_size: Number of pairs to sample. Required if use_sampling=True.

        Raises:
            ValueError: If sample_size is None when use_sampling=True.
        """
        if use_sampling and sample_size is None:
            raise ValueError("`sample_size` is required when `use_sampling=True`.")

        target_funcs = self._resolve_funcs(func_names)

        parallel_funcs = [f for f in target_funcs if not isinstance(f, BatchSimilarityFunction)]
        batch_funcs    = [f for f in target_funcs if isinstance(f, BatchSimilarityFunction)]

        if parallel_funcs:
            self._run_parallel_group(parallel_funcs, items, use_sampling, sample_size)

        for fn in batch_funcs:
            self._run_batch_function(fn, items, use_sampling, sample_size)

    def get(
        self,
        item_a: Dict[str, Any],
        item_b: Dict[str, Any],
    ) -> Dict[Tuple, float]:
        """Retrieve all cached similarities between two item dicts.

        For each registered function whose required key(s) are present in both
        dicts, looks up the cached similarity. If not cached, computes it on
        the fly (useful for small tests; use compute_all for large batches).

        Args:
            item_a: First item dict.
            item_b: Second item dict.

        Returns:
            Dict mapping (func.name, func.key_or_keys) to similarity float.
            Only functions whose required fields are present in both dicts
            are included. MultiKeySimilarityFunction keys are tuples of strings.

        Example:
            engine.get(
                {"produto_base": "leite", "ncm": "04011000"},
                {"produto_base": "leite condensado"},
            )
            # ncm excluded — item_b missing "ncm"
            # {("bert__all-MiniLM-L6-v2", "produto_base"): 0.68,
            #  ("sequence_match", "produto_base"): 0.61}
        """
        result: Dict[Tuple, float] = {}

        for fn in self._funcs.values():
            if isinstance(fn, MultiKeySimilarityFunction):
                if not all(k in item_a and k in item_b for k in fn.keys):
                    continue
                dict_key = (fn.name, tuple(fn.keys))
            else:
                if fn.key not in item_a or fn.key not in item_b:
                    continue
                dict_key = (fn.name, fn.key)

            val_a = fn.extract(item_a)
            val_b = fn.extract(item_b)
            cache_key = self._cache_key(fn, val_a, val_b)

            cached = self._cache.get(cache_key)
            if cached is not None:
                result[dict_key] = cached
            else:
                if isinstance(fn, BatchSimilarityFunction):
                    score = fn.compute_batch([(val_a, val_b)])[0]
                else:
                    score = fn.compute(val_a, val_b)
                self._cache.put(cache_key, float(score))
                result[dict_key] = float(score)

        return result

    def registered_functions(self) -> List[str]:
        """Return the names of all registered similarity functions.

        Returns:
            List of function name strings.
        """
        return list(self._funcs.keys())

    # ------------------------------------------------------------------
    # Cache key — single authoritative method
    # ------------------------------------------------------------------

    def _cache_key(self, fn: SimilarityFunction, val_a: Any, val_b: Any) -> bytes:
        """Generate a deterministic, opaque cache key for a similarity pair.

        The key encodes the function name and both extracted values. It is
        symmetric: (val_a, val_b) and (val_b, val_a) produce the same key.
        Stable across sessions (MD5, not Python hash()).

        Key format:
            MD5("{fn.name}|{repr(lo)}|{repr(hi)}")
            where lo/hi are sorted lexicographically for symmetry.

        This is the single source of truth for key generation. Never construct
        cache keys manually — always call this method.

        Args:
            fn: Similarity function (provides fn.name for namespacing).
            val_a: Extracted value for item A.
            val_b: Extracted value for item B.

        Returns:
            16-byte MD5 digest usable as KVStore SINGLE_KEY bytes key.
        """
        import hashlib
        rep_a = repr(val_a)
        rep_b = repr(val_b)
        lo, hi = (rep_a, rep_b) if rep_a <= rep_b else (rep_b, rep_a)
        raw = f"{fn.name}|{lo}|{hi}".encode("utf-8")
        return hashlib.md5(raw).digest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_funcs(self, func_names: Optional[List[str]]) -> List[SimilarityFunction]:
        """Resolve function names to SimilarityFunction instances.

        Args:
            func_names: Names to resolve. None means all registered.

        Returns:
            List of SimilarityFunction instances.

        Raises:
            KeyError: If a requested name is not registered.
        """
        if func_names is None:
            return list(self._funcs.values())
        result = []
        for name in func_names:
            if name not in self._funcs:
                raise KeyError(
                    f"Similarity function '{name}' is not registered. "
                    f"Available: {list(self._funcs.keys())}"
                )
            result.append(self._funcs[name])
        return result

    def _pairs_generator(
        self,
        n: int,
        use_sampling: bool,
        sample_size: Optional[int],
    ):
        """Generate index pairs (i, j) with i < j, optionally sampled.

        Uses reservoir sampling to avoid materializing all pairs when sampling.

        Args:
            n: Number of items (generates pairs over range(n)).
            use_sampling: If True, yield only a random sample.
            sample_size: Number of pairs to yield when sampling.

        Yields:
            Tuples (i, j) with i < j.
        """
        import math, random
        total = math.comb(n, 2)

        if not use_sampling or sample_size is None or sample_size >= total:
            yield from combinations(range(n), 2)
            return

        # Reservoir sampling over the combinations generator
        reservoir: List[Tuple[int, int]] = []
        for idx, pair in enumerate(combinations(range(n), 2)):
            if idx < sample_size:
                reservoir.append(pair)
            else:
                j = random.randint(0, idx)
                if j < sample_size:
                    reservoir[j] = pair

        yield from reservoir

    def _run_parallel_group(
        self,
        parallel_funcs: List[SimilarityFunction],
        items: List[Any],
        use_sampling: bool,
        sample_size: Optional[int],
    ) -> None:
        """Run all non-batch similarity functions in a single shared executor.

        For each function, pairs are streamed from a generator. Each cache miss
        is submitted to the executor immediately — workers process tasks while
        the main thread continues scanning for more missing pairs. Results are
        flushed to cache in batches of self._batch_size.

        Args:
            parallel_funcs: Non-batch SimilarityFunction instances to run.
            items: Full item dicts.
            use_sampling: Whether to sample pairs.
            sample_size: Number of pairs to sample per function.
        """
        import math

        # Populate registry once for all parallel functions
        global _FUNCTION_REGISTRY
        for fn in parallel_funcs:
            _FUNCTION_REGISTRY[fn.name] = fn

        MAX_INFLIGHT = self._max_workers * 200

        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            for fn in parallel_funcs:
                extracted_unique = list({fn.extract(item) for item in items})
                n_unique = len(extracted_unique)
                total_pairs = math.comb(n_unique, 2)

                print(
                    f"[SimilarityEngine] '{fn.name}': "
                    f"{n_unique:,} unique values, {total_pairs:,} pairs.",
                    flush=True,
                )

                pairs_gen = self._pairs_generator(n_unique, use_sampling, sample_size)

                # future -> (val_a, val_b) for cache key reconstruction
                inflight: Dict[Any, Tuple[Any, Any]] = {}
                writes: List[Tuple[bytes, float]] = []
                submitted = 0
                cached_count = 0

                def _flush_done() -> None:
                    done = [f for f in list(inflight) if f.done()]
                    for future in done:
                        val_a, val_b = inflight.pop(future)
                        score = future.result()
                        key = self._cache_key(fn, val_a, val_b)
                        writes.append((key, score))
                    if len(writes) >= self._batch_size:
                        self._cache.put_many(writes)
                        writes.clear()

                with tqdm(total=total_pairs, desc=fn.name) as pbar:
                    for i, j in pairs_gen:
                        val_a = extracted_unique[i]
                        val_b = extracted_unique[j]
                        pbar.update(1)

                        if self._cache.exists(self._cache_key(fn, val_a, val_b)):
                            cached_count += 1
                            continue

                        future = executor.submit(
                            _worker_compute,
                            (fn.name, val_a, val_b),
                        )
                        inflight[future] = (val_a, val_b)
                        submitted += 1

                        # drain done futures when inflight queue is full
                        if len(inflight) >= MAX_INFLIGHT:
                            _flush_done()

                    # drain remaining inflight
                    for future in as_completed(inflight):
                        val_a, val_b = inflight[future]
                        score = future.result()
                        key = self._cache_key(fn, val_a, val_b)
                        writes.append((key, score))

                if writes:
                    self._cache.put_many(writes)

                print(
                    f"[SimilarityEngine] '{fn.name}': "
                    f"{submitted:,} computed, {cached_count:,} already cached.",
                    flush=True,
                )

    def _run_batch_function(
        self,
        fn: BatchSimilarityFunction,
        items: List[Any],
        use_sampling: bool,
        sample_size: Optional[int],
    ) -> None:
        """Run a BatchSimilarityFunction over all pairs, chunked to limit memory.

        Pairs are streamed from a generator. Missing pairs are accumulated into
        chunks of self._batch_size, then sent to compute_batch and written to
        cache before the next chunk is processed.

        Args:
            fn: Batch similarity function to run.
            items: Full item dicts.
            use_sampling: Whether to sample pairs.
            sample_size: Number of pairs to sample.
        """
        import math

        extracted_unique = list({fn.extract(item) for item in items})
        n_unique = len(extracted_unique)
        total_pairs = math.comb(n_unique, 2)

        print(
            f"[SimilarityEngine] '{fn.name}': "
            f"{n_unique:,} unique values, {total_pairs:,} pairs.",
            flush=True,
        )

        pairs_gen = self._pairs_generator(n_unique, use_sampling, sample_size)

        chunk: List[Tuple[int, int]] = []
        cached_count = 0
        computed = 0

        with tqdm(total=total_pairs, desc=fn.name) as pbar:
            for i, j in pairs_gen:
                pbar.update(1)
                key = self._cache_key(fn, extracted_unique[i], extracted_unique[j])
                if self._cache.exists(key):
                    cached_count += 1
                    continue
                chunk.append((i, j))

                if len(chunk) >= self._batch_size:
                    self._flush_batch_chunk(fn, extracted_unique, chunk)
                    computed += len(chunk)
                    chunk = []

        if chunk:
            self._flush_batch_chunk(fn, extracted_unique, chunk)
            computed += len(chunk)

        print(
            f"[SimilarityEngine] '{fn.name}': "
            f"{computed:,} computed, {cached_count:,} already cached.",
            flush=True,
        )

    def _flush_batch_chunk(
        self,
        fn: BatchSimilarityFunction,
        extracted_unique: List[Any],
        chunk: List[Tuple[int, int]],
    ) -> None:
        """Send one chunk of missing pairs to compute_batch and write to cache.

        Args:
            fn: Batch similarity function.
            extracted_unique: Deduplicated extracted values.
            chunk: List of (i, j) index pairs to compute.
        """
        pairs = [(extracted_unique[i], extracted_unique[j]) for i, j in chunk]
        scores = fn.compute_batch(pairs)
        writes = [
            (self._cache_key(fn, extracted_unique[i], extracted_unique[j]), float(score))
            for (i, j), score in zip(chunk, scores)
        ]
        self._cache.put_many(writes)

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

    def test_base_name_and_key() -> None:
        """Test that name is generated correctly from base_name and key."""
        print("\n[test_base_name_and_key]")
        fn1 = SequenceMatchSimilarity("produto_base")
        fn2 = SequenceMatchSimilarity("produto_detalhado")
        _assert(fn1.name == "sequence_match__produto_base", "name includes key")
        _assert(fn2.name == "sequence_match__produto_detalhado", "different keys -> different names")
        _assert(fn1.name != fn2.name, "no cache collision between different keys")

    def test_extract_single_key() -> None:
        """Test extract() for single-key functions."""
        print("\n[test_extract_single_key]")
        fn = SequenceMatchSimilarity("produto_base")
        item = {"produto_base": "Agua Mineral", "ncm": "22011000"}
        _assert(fn.extract(item) == "Agua Mineral", "extracts correct field")
        _assert(fn.extract({"other": "x"}) is None, "missing key returns None")
        _assert(fn.extract("raw string") == "raw string", "non-dict passthrough")

    def test_extract_multi_key() -> None:
        """Test extract() for multi-key functions."""
        print("\n[test_extract_multi_key]")
        fn = NumericRangeSimilarity("valor", "unidade", 0.0, 100.0)
        item = {"valor": 42.0, "unidade": "UN", "other": "x"}
        result = fn.extract(item)
        _assert(result == {"valor": 42.0, "unidade": "UN"}, "extracts only declared keys")
        _assert(fn.extract({}) == {"valor": None, "unidade": None}, "missing keys return None")

    def test_sequence_match() -> None:
        """Test SequenceMatchSimilarity."""
        print("\n[test_sequence_match]")
        fn = SequenceMatchSimilarity("k")
        _assert(fn.compute("agua mineral", "agua mineral") == 1.0, "identical -> 1.0")
        _assert(fn.compute("AGUA MINERAL", "agua mineral") == 1.0, "case insensitive")
        _assert(fn.compute("agua mineral", "suco laranja") < 0.5, "unrelated -> < 0.5")
        _assert(fn.compute(None, "agua") == 0.0, "None -> 0.0")

    def test_ncm_similarity() -> None:
        """Test NCMSimilarity prefix matching and zero-padding."""
        print("\n[test_ncm_similarity]")
        fn = NCMSimilarity("ncm")
        _assert(fn.compute("01012100", "01012100") == 1.0, "identical -> 1.0")
        _assert(fn.compute("01012100", "01012200") > 0.5, "shared prefix -> > 0.5")
        _assert(fn.compute("1012100", "01012100") == 1.0, "zero-padded to 8 digits")
        _assert(fn.compute(None, "01012100") == 0.0, "None -> 0.0")

    def test_category_similarity() -> None:
        """Test CategorySimilarity Jaccard."""
        print("\n[test_category_similarity]")
        fn = CategorySimilarity("categoria")
        _assert(fn.compute(["Bebidas", "água"], ["Bebidas", "água"]) == 1.0, "identical -> 1.0")
        _assert(fn.compute(["Bebidas"], ["Alimentos"]) == 0.0, "no overlap -> 0.0")
        _assert(0.0 < fn.compute(["Bebidas", "água"], ["Bebidas", "suco"]) < 1.0, "partial")

    def test_tag_similarity() -> None:
        """Test TagSimilarity Jaccard."""
        print("\n[test_tag_similarity]")
        fn = TagSimilarity("tags")
        _assert(fn.compute(["agua", "500ml"], ["agua", "500ml"]) == 1.0, "identical -> 1.0")
        _assert(fn.compute(["agua"], ["suco"]) == 0.0, "no overlap -> 0.0")

    def test_numeric_range_similarity() -> None:
        """Test NumericRangeSimilarity with and without group."""
        print("\n[test_numeric_range_similarity]")
        fn = NumericRangeSimilarity("valor", "unidade", 0.0, 100.0)
        _assert(
            fn.compute({"valor": 50.0, "unidade": "UN"}, {"valor": 50.0, "unidade": "UN"}) == 1.0,
            "same -> 1.0",
        )
        _assert(
            fn.compute({"valor": 10.0, "unidade": "UN"}, {"valor": 10.0, "unidade": "KG"}) == 0.0,
            "diff group -> 0.0",
        )
        _assert(
            fn.compute({"valor": 0.0, "unidade": None}, {"valor": 100.0, "unidade": None}) == 0.0,
            "max dist -> 0.0",
        )
        _assert(
            0.0 < fn.compute({"valor": 40.0, "unidade": None}, {"valor": 60.0, "unidade": None}) < 1.0,
            "partial",
        )
        fn_no_group = NumericRangeSimilarity("valor", None, 0.0, 100.0)
        _assert(fn_no_group.name == "numeric_range__valor", "no group -> single key in name")

    def test_bert_similarity() -> None:
        """Test BERTSimilarity compute and compute_batch consistency."""
        print("\n[test_bert_similarity]")
        try:
            fn = BERTSimilarity("produto_base")
        except ImportError:
            print("  SKIP: sentence-transformers not installed.")
            return
        _assert(fn.name == "bert__produto_base", "name includes key")
        sim = fn.compute("agua mineral", "agua mineral")
        _assert(sim > 0.99, "identical -> close to 1.0")
        batch = fn.compute_batch([
            ("agua mineral", "agua mineral"),
            ("agua mineral", "suco de laranja"),
        ])
        _assert(batch[0] > batch[1], "identical > unrelated")
        _assert(all(0.0 <= s <= 1.0 for s in batch), "scores in [0,1]")
        _assert(fn.compute(None, "agua") == 0.0, "None -> 0.0")

    def test_multikey_isinstance() -> None:
        """Test that MultiKeyBatchSimilarityFunction is detected as BatchSimilarityFunction."""
        print("\n[test_multikey_isinstance]")

        class DummyMultiKeyBatch(MultiKeyBatchSimilarityFunction):
            base_name = "dummy_mkb"

            def compute(self, a, b):
                return 1.0

            def compute_batch(self, pairs):
                return [1.0] * len(pairs)

        fn = DummyMultiKeyBatch(["k1", "k2"])
        _assert(isinstance(fn, BatchSimilarityFunction), "detected as BatchSimilarityFunction")
        _assert(isinstance(fn, MultiKeySimilarityFunction), "detected as MultiKeySimilarityFunction")
        _assert(fn.name == "dummy_mkb__k1,k2", "name includes all keys")
        _assert(fn.extract({"k1": "x", "k2": "y", "z": "!"}) == {"k1": "x", "k2": "y"}, "multi extract")

    def test_engine_with_dict_items() -> None:
        """Test SimilarityEngine end-to-end with dict items and multiple functions."""
        print("\n[test_engine_with_dict_items]")

        class MockCache:
            def __init__(self):
                self._data = {}
            def exists(self, key):
                return key in self._data
            def get(self, key, default=None):
                return self._data.get(key, default)
            def put_many(self, items):
                for k, v in items:
                    self._data[k] = v

        items = [
            {"produto_base": "agua mineral", "tags": ["agua", "mineral"]},
            {"produto_base": "agua com gas",  "tags": ["agua", "gas"]},
            {"produto_base": "suco laranja",  "tags": ["suco", "laranja"]},
        ]
        ids = [1, 2, 3]

        engine = SimilarityEngine(
            funcs=[
                SequenceMatchSimilarity("produto_base"),
                TagSimilarity("tags"),
            ],
            cache=MockCache(),
            max_workers=2,
        )
        engine.compute_all(items=items, ids=ids)

        sim_seq_12 = engine.get("sequence_match__produto_base", 1, 2)
        sim_seq_13 = engine.get("sequence_match__produto_base", 1, 3)
        sim_tag_12 = engine.get("tags__tags", 1, 2)
        sim_tag_13 = engine.get("tags__tags", 1, 3)

        _assert(sim_seq_12 is not None, "sequence pair (1,2) cached")
        _assert(sim_seq_13 is not None, "sequence pair (1,3) cached")
        _assert(sim_seq_12 > sim_seq_13, "agua pairs more similar than agua/suco")
        _assert(sim_tag_12 > sim_tag_13, "shared tag agua makes (1,2) more similar")

        # Idempotency
        engine.compute_all(items=items, ids=ids)
        _assert(engine.get("sequence_match__produto_base", 1, 2) == sim_seq_12, "idempotent")

        # Duplicate name raises
        try:
            SimilarityEngine(
                funcs=[SequenceMatchSimilarity("k"), SequenceMatchSimilarity("k")],
                cache=MockCache(),
            )
            _assert(False, "should have raised ValueError")
        except ValueError:
            _assert(True, "duplicate name raises ValueError")

    print("=" * 60)
    print("similarity.py — running tests")
    print("=" * 60)

    test_base_name_and_key()
    test_extract_single_key()
    test_extract_multi_key()
    test_sequence_match()
    test_ncm_similarity()
    test_category_similarity()
    test_tag_similarity()
    test_numeric_range_similarity()
    test_bert_similarity()
    test_multikey_isinstance()
    test_engine_with_dict_items()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)