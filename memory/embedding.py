"""Embedding model using Microsoft's harrier-oss-v1 with SQLite caching."""

import os
import sqlite3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Optional

# Shared layered config (config.yaml < secrets < env vars)
import memory_config as cfg

# HuggingFace token for faster downloads and higher rate limits
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Available models
MODELS = {
    "270m": {
        "name": "microsoft/harrier-oss-v1-270m",
        "dimension": 640,
    },
    "0.6b": {
        "name": "microsoft/harrier-oss-v1-0.6b",
        "dimension": 1024,
    },
    "27b": {
        "name": "microsoft/harrier-oss-v1-27b",
        "dimension": 5376,
    },
}

# Model settings from config (defaults)
DEFAULT_MODEL_SIZE = cfg.get('embedding.model_size', '270m')
FORCE_CPU = cfg.get('embedding.force_cpu', True)

# Cache settings from config (paths resolved relative to this file)
_CACHE_DIR_DEFAULT = cfg.get('cache.cache_dir', 'database')
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), _CACHE_DIR_DEFAULT)
os.makedirs(CACHE_DIR, exist_ok=True)
DEFAULT_CACHE_DB = os.path.join(CACHE_DIR, cfg.get('cache.cache_db', 'embeddings_cache.db'))


class EmbeddingModel:
    """Harrier embedding model with SQLite caching."""
    
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        cache_db: str = DEFAULT_CACHE_DB,
        device: str | None = None,
    ):
        self.model_size = model_size
        self.model_name = MODELS[model_size]["name"]
        self.dimension = MODELS[model_size]["dimension"]
        self.cache_db = cache_db
        
        # Determine device
        if device is not None:
            self.device = device
        elif FORCE_CPU:
            self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Load model
        print(f"Loading {self.model_name} on {self.device}...")
        try:
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                token=HF_TOKEN
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"Warning: CUDA failed ({e}), falling back to CPU")
                self.device = "cpu"
                self.model = SentenceTransformer(self.model_name, device="cpu")
            else:
                raise
        
        print(f"Loaded {self.model_name} (dim={self.dimension})")
        self._init_cache()
    
    def _init_cache(self):
        """Initialize the cache database."""
        # Ensure directory exists
        import os
        cache_dir = os.path.dirname(self.cache_db)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text TEXT PRIMARY KEY,
                    embedding BLOB
                )
            """)
    
    def get(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding as numpy array (float32)
        """
        text_lower = text.lower().strip()
        if not text_lower:
            return np.zeros(self.dimension, dtype=np.float32)
        
        # Check cache first
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT embedding FROM embeddings WHERE text = ?",
                    (text_lower,)
                )
                row = cursor.fetchone()
                
                if row:
                    return np.frombuffer(row[0], dtype=np.float32)
        except Exception:
            pass  # If cache fails, generate new embedding
        
        # Generate new embedding
        embedding = self.model.encode(text_lower, normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)
        
        # Cache it
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (text, embedding) VALUES (?, ?)",
                    (text_lower, embedding.tobytes())
                )
        except Exception:
            pass  # If cache fails, continue anyway
        
        return embedding
    
    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode multiple texts.
        
        Args:
            texts: List of texts to encode
            normalize: L2 normalize embeddings
            
        Returns:
            Embeddings as numpy array
        """
        return self.model.encode(texts, normalize_embeddings=normalize)
    
    def clear_cache(self) -> int:
        """Clear the embedding cache.
        
        Returns:
            Number of entries deleted
        """
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("DELETE FROM embeddings")
                conn.commit()
                return cursor.rowcount
        except Exception:
            return 0
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dict with cache info
        """
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]
                return {"count": count, "cache_db": self.cache_db}
        except Exception as e:
            return {"error": str(e)}


# Default instance
_default_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    model_size: str = DEFAULT_MODEL_SIZE,
    cache_db: str = DEFAULT_CACHE_DB,
    device: str | None = None,
) -> EmbeddingModel:
    """Get or create the default embedding model."""
    global _default_model
    
    if _default_model is None:
        _default_model = EmbeddingModel(
            model_size=model_size,
            cache_db=cache_db,
            device=device,
        )
    
    return _default_model


# Backwards compatibility alias
DEFAULT_MODEL = MODELS[DEFAULT_MODEL_SIZE]["name"]


if __name__ == "__main__":
    # Test the default model (27b with CPU)
    print(f"Using model: {DEFAULT_MODEL_SIZE}, FORCE_CPU: {FORCE_CPU}")
    model = EmbeddingModel()
    
    # Test encoding
    text = "Hello world, this is a test embedding"
    emb = model.get(text)
    print(f"Single embedding shape: {emb.shape}")
    print(f"Sample values: {emb[:5]}")
    
    # Test batch
    texts = ["test one", "test two", "test three"]
    embs = model.encode(texts)
    print(f"Batch embedding shape: {embs.shape}")
    
    # Test cache
    emb2 = model.get(text)
    print(f"Cache working: {np.allclose(emb, emb2)}")
    
    # Stats
    stats = model.get_cache_stats()
    print(f"Cache stats: {stats}")