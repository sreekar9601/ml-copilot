"""Text embedding module using Nomic AI embedding model."""

import logging
from typing import List, Dict, Any
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class NomicEmbedder:
    """Text embedder using Nomic AI's embedding model."""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """Load the tokenizer and model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode a batch of texts into embeddings."""
        if self.model is None:
            self.load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                    
                    # Normalize embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy and move to CPU
                batch_embeddings = embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)
                
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Create zero embeddings for failed batch
                batch_size_actual = len(batch_texts)
                embedding_dim = 768  # Default for nomic-embed-text-v1
                zero_embeddings = np.zeros((batch_size_actual, embedding_dim))
                all_embeddings.append(zero_embeddings)
        
        # Concatenate all embeddings
        final_embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated embeddings for {len(texts)} texts, shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding."""
        return self.encode_batch([text])[0]
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query text with special formatting for retrieval."""
        # For Nomic embeddings, we can add a prefix to indicate this is a search query
        formatted_query = f"search_query: {query}"
        return self.encode_single(formatted_query)
    
    def encode_document(self, text: str) -> np.ndarray:
        """Encode a document text with special formatting."""
        # For document encoding, we can add a prefix
        formatted_text = f"search_document: {text}"
        return self.encode_single(formatted_text)


class EmbeddingCache:
    """Simple in-memory cache for embeddings to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def get(self, text: str) -> np.ndarray:
        """Get embedding from cache."""
        if text in self.cache:
            # Move to end of access order
            self.access_order.remove(text)
            self.access_order.append(text)
            return self.cache[text]
        return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[text] = embedding
        self.access_order.append(text)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


class CachedEmbedder:
    """Embedder with caching capabilities."""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1", cache_size: int = 1000):
        self.embedder = NomicEmbedder(model_name)
        self.cache = EmbeddingCache(cache_size)
    
    def encode_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode batch with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            new_embeddings = self.embedder.encode_batch(uncached_texts, batch_size)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                self.cache.put(text, embedding)
            
            # Add to results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original order and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text with caching."""
        return self.encode_batch([text])[0]
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query with caching."""
        formatted_query = f"search_query: {query}"
        return self.encode_single(formatted_query)


if __name__ == "__main__":
    # Test the embedder
    logging.basicConfig(level=logging.INFO)
    
    embedder = NomicEmbedder()
    
    test_texts = [
        "How to use PyTorch DataLoader?",
        "MLflow model registry concepts",
        "KServe inference service deployment",
        "Ray Serve production configuration"
    ]
    
    embeddings = embedder.encode_batch(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity
    query_embedding = embedder.encode_query("PyTorch data loading")
    similarities = np.dot(embeddings, query_embedding)
    
    print("Similarities with query 'PyTorch data loading':")
    for text, sim in zip(test_texts, similarities):
        print(f"  {text}: {sim:.3f}")

