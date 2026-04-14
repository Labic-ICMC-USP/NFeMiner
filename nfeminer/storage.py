import os
import msgpack
import lmdb
import struct
import hashlib
from enum import IntEnum
from typing import Any, Optional, Iterable, Iterator,Tuple

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class KVStore:
    """
    High-performance persistent key-value store built on LMDB.

    Characteristics:
        - Fixed key layout per database
        - Strong durability guarantees (commit on every write call)
        - Metadata persisted internally
        - Optimized for large datasets
        - Deterministic binary encoding

    Each database instance is homogeneous:
        - Single KeyMode
        - Single ValueMode
        - Fixed key size
        - Fixed key arity
    """

    VERSION = 1
    _META_KEY = b"__meta__"

    class KeyMode(IntEnum):
        """Defines how keys are encoded."""
        MULTI_KEY = 1
        SINGLE_KEY = 2

    class ValueMode(IntEnum):
        """Defines how values are serialized."""
        NUMERIC = 1
        MSGPACK = 2

    def __init__(
        self,
        path: str,
        *,
        key_mode: Optional["KVStore.KeyMode"] = None,
        value_mode: Optional["KVStore.ValueMode"] = None,
        map_size_bytes: int = 2,
        map_size: int = 1 << 40,
        readonly: bool = False,
    ):
        """
        Initialize or open a KVStore.

        If database does not exist:
            - key_mode and value_mode are required.
            - Metadata is persisted.

        If database exists:
            - Configuration is loaded from metadata.

        Args:
            path: Filesystem path for LMDB storage.
            key_mode: Key encoding mode (required if creating DB).
            value_mode: Value serialization mode (required if creating DB).
            map_size_bytes: Bytes used for string mapping (1, 2, or 4).
            map_size: Maximum LMDB map size.
            readonly: Open database in read-only mode.

        Raises:
            ValueError: If configuration is invalid or inconsistent.
        """
        if map_size_bytes not in (1, 2, 4):
            raise ValueError("map_size_bytes must be 1, 2, or 4")

        self._path = path
        self._readonly = readonly
        self._map_size_bytes = map_size_bytes

        os.makedirs(path, exist_ok=True)

        self._env = lmdb.open(
            path,
            map_size=map_size,
            max_dbs=3,
            readonly=readonly,
            lock=not readonly,
            sync=True,
            metasync=True,
        )

        self._db_meta = self._env.open_db(b"meta")
        self._db_data = self._env.open_db(b"data")
        self._db_map = self._env.open_db(b"map")

        with self._env.begin(write=not readonly) as txn:
            raw_meta = txn.get(self._META_KEY, db=self._db_meta)

            if raw_meta is None:
                if readonly:
                    raise ValueError("Database does not exist and readonly=True")

                if key_mode is None or value_mode is None:
                    raise ValueError("key_mode and value_mode required when creating DB")

                self._key_mode = key_mode
                self._value_mode = value_mode
                self._key_size = None
                self._key_arity = None
                self._string_map_cache = {}
                self._string_map_counter = 0

                meta = {
                    "version": self.VERSION,
                    "key_mode": int(key_mode),
                    "value_mode": int(value_mode),
                    "map_size_bytes": map_size_bytes,
                    "key_size": None,
                    "key_arity": None,
                }

                txn.put(
                    self._META_KEY,
                    msgpack.packb(meta),
                    db=self._db_meta,
                )
            else:
                meta = msgpack.unpackb(raw_meta)

                self._key_mode = self.KeyMode(meta["key_mode"])
                self._value_mode = self.ValueMode(meta["value_mode"])
                self._map_size_bytes = meta["map_size_bytes"]
                self._key_size = meta["key_size"]
                self._key_arity = meta["key_arity"]
                self._string_map_cache, self._string_map_counter = self._load_string_map()

    # =========================
    # Public API
    # =========================

    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update a single key-value pair.

        This operation performs a fully committed transaction.
        Once this method returns successfully, data is durably stored.

        Args:
            key: Key to insert.
            value: Value to store.
        """
        encoded_key = self._encode_key(key)
        encoded_value = self._encode_value(value)

        with self._env.begin(write=True) as txn:
            txn.put(encoded_key, encoded_value, db=self._db_data)

    def put_many(self, items: Iterable[Tuple[Any, Any]]) -> None:
        """
        Insert or update multiple key-value pairs in a single transaction.

        This operation performs one committed transaction.

        Args:
            items: Iterable of (key, value) pairs.
        """
        with self._env.begin(write=True) as txn:
            for key, value in items:
                encoded_key = self._encode_key(key)
                encoded_value = self._encode_value(value)
                txn.put(encoded_key, encoded_value, db=self._db_data)

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Retrieve a value by key.

        Args:
            key: Key to retrieve.
            default: Value returned if key does not exist.

        Returns:
            Deserialized value or default.
        """
        encoded_key = self._encode_key(key)

        with self._env.begin() as txn:
            raw = txn.get(encoded_key, db=self._db_data)
            if raw is None:
                return default
            return self._decode_value(raw)

    def exists(self, key: Any) -> bool:
        """
        Check whether a key exists.

        Args:
            key: Key to check.

        Returns:
            True if key exists, False otherwise.
        """
        encoded_key = self._encode_key(key)
        with self._env.begin() as txn:
            return txn.get(encoded_key, db=self._db_data) is not None

    def delete(self, key: Any) -> None:
        """
        Remove a key from the store.

        Args:
            key: Key to remove.
        """
        encoded_key = self._encode_key(key)
        with self._env.begin(write=True) as txn:
            txn.delete(encoded_key, db=self._db_data)

    def items(self, prefix: Any = None) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over key-value pairs.

        Args:
            prefix: Optional key prefix for range scan.

        Yields:
            Tuple of (raw_key_bytes, deserialized_value).
        """
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._db_data)

            if prefix is None:
                for k, v in cursor:
                    yield k, self._decode_value(v)
            else:
                encoded_prefix = self._encode_key(prefix)
                if cursor.set_range(encoded_prefix):
                    for k, v in cursor:
                        if not k.startswith(encoded_prefix):
                            break
                        yield k, self._decode_value(v)

    def info(self) -> dict:
        """
        Return metadata and configuration.

        Returns:
            Dictionary containing configuration details.
        """
        return {
            "version": self.VERSION,
            "key_mode": self._key_mode,
            "value_mode": self._value_mode,
            "key_size": self._key_size,
            "key_arity": self._key_arity,
            "map_size_bytes": self._map_size_bytes,
        }

    def close(self) -> None:
        """Close the LMDB environment."""
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================
    # Internal Encoding
    # =========================

    def _encode_key(self, key: Any) -> bytes:
        if self._key_mode == self.KeyMode.SINGLE_KEY:
            encoded = self._encode_single_key(key)
            arity = 1
        else:
            if not isinstance(key, tuple):
                raise ValueError("MULTI_KEY requires tuple")
            encoded = self._encode_multi_key(key)
            arity = len(key)

        if self._key_size is None:
            self._key_size = len(encoded)
            self._key_arity = arity
            self._persist_meta()
        else:
            if len(encoded) != self._key_size:
                raise ValueError("Key size mismatch")
            if arity != self._key_arity:
                raise ValueError("Key arity mismatch")

        return encoded

    def _encode_single_key(self, key: Any) -> bytes:
        if isinstance(key, str):
            return hashlib.sha256(key.encode()).digest()
        if isinstance(key, (int, float)):
            return struct.pack(">d", float(key))
        if isinstance(key, bytes):
            return key
        raise ValueError("Unsupported key type")

    def _encode_multi_key(self, key: tuple) -> bytes:
        parts = []
        for item in key:
            if isinstance(item, str):
                parts.append(self._encode_string_map(item))
            elif isinstance(item, (int, float)):
                parts.append(struct.pack(">d", float(item)))
            else:
                raise ValueError("MULTI_KEY supports only str and numeric types")
        return b"".join(parts)

    def _encode_string_map(self, value: str) -> bytes:
        if value in self._string_map_cache:
            idx = self._string_map_cache[value]
        else:
            idx = self._string_map_counter
            max_value = (1 << (8 * self._map_size_bytes)) - 1
            if idx > max_value:
                raise ValueError("String map capacity exceeded")
            self._string_map_cache[value] = idx
            self._persist_string_mapping(value, idx)
            self._string_map_counter += 1

        return idx.to_bytes(self._map_size_bytes, "big")

    def _encode_value(self, value: Any) -> bytes:
        if self._value_mode == self.ValueMode.NUMERIC:
            return struct.pack(">d", float(value))
        return msgpack.packb(value)

    def _decode_value(self, raw: bytes) -> Any:
        if self._value_mode == self.ValueMode.NUMERIC:
            return struct.unpack(">d", raw)[0]
        return msgpack.unpackb(raw)

    # =========================
    # Meta Persistence
    # =========================

    def _persist_meta(self):
        meta = {
            "version": self.VERSION,
            "key_mode": int(self._key_mode),
            "value_mode": int(self._value_mode),
            "map_size_bytes": self._map_size_bytes,
            "key_size": self._key_size,
            "key_arity": self._key_arity,
        }
        with self._env.begin(write=True) as txn:
            txn.put(self._META_KEY, msgpack.packb(meta), db=self._db_meta)

    def _persist_string_mapping(self, value: str, idx: int):
        with self._env.begin(write=True) as txn:
            txn.put(value.encode(), idx.to_bytes(self._map_size_bytes, "big"), db=self._db_map)

    def _load_string_map(self) -> tuple[dict, int]:
        cache = {}
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._db_map)
            for k, v in cursor:
                string_val = k.decode()
                idx = int.from_bytes(v, "big")
                cache[string_val] = idx
        return cache, len(cache)

if __name__ == "__main__":
    print("\n==============================")
    print("KVSTORE TESTS")
    print("==============================")

    KV_PATH = "kv_test"

    if os.path.exists(KV_PATH):
        import shutil
        shutil.rmtree(KV_PATH)

    print("\n--- Creating KVStore ---")

    kv = KVStore(
        KV_PATH,
        key_mode=KVStore.KeyMode.SINGLE_KEY,
        value_mode=KVStore.ValueMode.NUMERIC,
    )

    print("\n--- Single Put ---")
    kv.put(1, 3.14)
    print("Value for key 1:", kv.get(1))

    print("\n--- Batch Put ---")
    kv.put_many([
        (2, 2.71),
        (3, 1.41),
        (4, 0.577),
        (5, 8),
    ])

    print("Value for key 2:", kv.get(2))
    print("Value for key 3:", kv.get(3))

    print("\n--- Exists Test ---")
    print("Key 1 exists?", kv.exists(1))
    print("Key 99 exists?", kv.exists(99))

    print("\n--- Delete Test ---")
    kv.delete(1)
    print("Key 1 exists after delete?", kv.exists(1))

    print("\n--- Iteration Test ---")
    for k, v in kv.items():
        print("Key:", k, "Value:", v)

    kv.close()

    print("\n==============================")
    print("KVSTORE TESTS FINISHED")
    print("==============================")