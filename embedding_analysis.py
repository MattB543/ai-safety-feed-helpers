import os
import logging
import json
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import psycopg2
from collections import Counter
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# OpenAI and BERTopic for labeling
import openai
from openai import OpenAI
from bertopic import BERTopic
from bertopic.representation import OpenAI as BERTopicOpenAI
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer

# Add tenacity for retry logic
from tenacity import retry, wait_random_exponential, stop_after_attempt

"""
Advanced Embedding Analysis with OpenAI Integration

This module provides comprehensive embedding analysis capabilities with:
- Direct PostgreSQL integration for loading pre-computed embeddings
- UMAP dimensionality reduction: high-dimensional (15D) for clustering, 2D for visualization
- HDBSCAN clustering optimized for high-dimensional embeddings
- OpenAI embeddings integration via custom adapter class
- BERTopic topic modeling with multiple representation models
- GPT-4 powered cluster labeling
- Interactive visualizations

Key Features:
- High-dimensional clustering: Uses 15D UMAP embeddings for clustering to preserve structure
- 2D visualization: Separate 2D UMAP projection for visualization only
- OpenAIEmbedder: Custom adapter making OpenAI embeddings compatible with BERTopic/KeyBERT/MMR
- Consistent embedding models throughout the pipeline (database → BERTopic → representations)
- Fallback to SentenceTransformer when OpenAI is unavailable
- Comprehensive outlier analysis and visualization

Clustering Strategy:
- Original embeddings (512D-1536D) → UMAP to 15D → HDBSCAN clustering
- Original embeddings → UMAP to 2D → Visualization only
- This approach avoids the structural distortions that can occur when clustering in 2D

Dependencies:
- openai>=1.3.8
- bertopic
- sentence-transformers  
- tenacity
- umap-learn
- hdbscan
- plotly
- psycopg2
- scikit-learn
- pandas
- numpy

Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key
- AI_SAFETY_FEED_DB_URL: PostgreSQL connection string
"""

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """
    Minimal adapter that makes OpenAI embeddings look like a SentenceTransformer.
    Works with BERTopic/KeyBERT/MMR out-of-the-box.
    """
    def __init__(self,
                 model: str = "text-embedding-3-small",
                 dims: int = 1536,             # 512 or 1536
                 batch_size: int = 100,
                 client: openai.OpenAI | None = None):
        self.model = model
        self.dims = dims
        self.batch_size = batch_size
        self.client = client or OpenAI()       # uses env var OPENAI_API_KEY

    # ---------- required by BERTopic rep-models ----------
    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(6))
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dims
        )
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[i:i+self.batch_size]))
        return np.vstack(vectors)

    # optional but nice to have
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.embed_documents(queries)

    # KeyBERT compatibility - KeyBERTInspired calls .encode() on the model
    def encode(self, texts, **kwargs):
        """
        KeyBERT-compatible encoding method.
        KeyBERTInspired expects this interface.
        """
        return self.embed_documents(texts)


class EmbeddingAnalyzer:
    """
    A comprehensive class for analyzing embeddings with clustering and GPT-4 labeling.
    
    This implementation follows 2025 best practices for embedding analysis:
    - Uses environment variables for secure credential management
    - Implements both direct OpenAI API calls and BERTopic integration
    - Provides comprehensive outlier analysis
    - Supports multiple clustering visualization options
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 db_url: Optional[str] = None,
                 random_state: int = 42):
        """
        Initialize the analyzer with database and OpenAI configurations.
        
        Args:
            openai_api_key: OpenAI API key (defaults to environment variable)
            db_url: Database URL (defaults to environment variable)
            random_state: Random seed for reproducibility (default: 42)
        """
        # Set random seed for reproducibility
        np.random.seed(random_state)
        self.random_state = random_state
        
        # Setup OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=openai_api_key or os.getenv("OPEN_AI_FREE_CREDITS_KEY")
        )
        
        # Database URL from environment variable or provided url
        self.db_url = db_url or os.getenv("AI_SAFETY_FEED_DB_URL")
        
        if not self.db_url:
            raise ValueError("Database URL not provided. Set AI_SAFETY_FEED_DB_URL environment variable.")
        
        # Initialize components
        self.embeddings = None
        self.ids = None
        self.titles = None
        self.embedding_umap = None
        self.embedding_2d = None
        self.cluster_labels = None
        self.outlier_scores = None
        self.topic_model = None
        
    def _embedding_to_array(self, embedding) -> np.ndarray:
        """
        Convert various embedding formats to numpy array.
        
        Handles different pgvector formats:
        - memoryview (common with psycopg2 and pgvector)
        - string representations like "[0.1,0.2]"
        - lists, tuples, and other array-like objects
        
        Args:
            embedding: Embedding in various formats
            
        Returns:
            numpy array of float32 values
        """
        if isinstance(embedding, memoryview):
            return np.frombuffer(embedding, dtype=np.float32).copy()
        if isinstance(embedding, str):
            return np.array(literal_eval(embedding), dtype=np.float32)
        return np.asarray(embedding, dtype=np.float32)
        
    def load_embeddings_from_db(self, 
                               query: Optional[str] = None,
                               limit: Optional[int] = None) -> Tuple[List[int], List[str], np.ndarray]:
        """
        Load embeddings from PostgreSQL database.
        
        Args:
            query: Custom SQL query (optional)
            limit: Limit number of records (optional)
            
        Returns:
            Tuple of (ids, titles, embeddings)
        """
        if query is None:
            base_query = "SELECT id, cleaned_title, embedding_short FROM content"
            if limit:
                query = f"{base_query} LIMIT {limit}"
            else:
                query = base_query
                
        logger.info(f"Connecting to database: {self.db_url}")
        
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    records = cur.fetchall()
                    
                    # Separate the results
                    self.ids = []
                    self.titles = []
                    embeddings_list = []
                    
                    for record in records:
                        article_id, title, embedding = record
                        self.ids.append(article_id)
                        self.titles.append(title or f"Article {article_id}")  # Handle None titles
                        
                        # Convert embedding to numeric array using robust conversion
                        embeddings_list.append(self._embedding_to_array(embedding))
                    
                    # Convert to numpy array
                    self.embeddings = np.array(embeddings_list)
                    
                    logger.info(f"Loaded {self.embeddings.shape[0]} embeddings with dimension {self.embeddings.shape[1]}")
                    return self.ids, self.titles, self.embeddings
                    
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def reduce_dimensionality(self, 
                            n_components: int = 15,  # Changed default to 15 for clustering
                            n_neighbors: int = 30,   # Increased from 15 to 30 for denser manifold
                            min_dist: float = 0.1,
                            metric: str = 'cosine',
                            create_2d_for_viz: bool = True) -> np.ndarray:
        """
        Apply UMAP for dimensionality reduction.
        
        Args:
            n_components: Number of dimensions to reduce to (default: 15 for clustering)
            n_neighbors: UMAP parameter for local neighborhood size
            min_dist: UMAP parameter for minimum distance between points
            metric: Distance metric to use
            create_2d_for_viz: Whether to also create 2D embeddings for visualization
            
        Returns:
            Reduced embeddings array
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings_from_db() first.")
            
        logger.info(f"Reducing {self.embeddings.shape[1]}D embeddings to {n_components}D using UMAP")
        
        # Initialize UMAP with specified parameters
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state
        )
        
        # Fit and transform the embeddings
        self.embedding_umap = umap_reducer.fit_transform(self.embeddings)
        
        # Create 2D embeddings for visualization if requested and not already 2D
        if create_2d_for_viz and n_components != 2:
            logger.info("Creating additional 2D embeddings for visualization...")
            umap_2d = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=self.random_state
            )
            self.embedding_2d = umap_2d.fit_transform(self.embeddings)
        elif n_components == 2:
            self.embedding_2d = self.embedding_umap
        
        logger.info(f"UMAP reduction complete. Shape: {self.embedding_umap.shape}")
        if hasattr(self, 'embedding_2d') and self.embedding_2d is not None:
            logger.info(f"2D embeddings for visualization: {self.embedding_2d.shape}")
        
        return self.embedding_umap
    
    def cluster_embeddings(self,
                          min_cluster_size: int = 5,
                          min_samples: int = 5,
                          metric: str = 'euclidean',
                          cluster_selection_method: str = 'leaf',
                          cluster_selection_epsilon: float = 0.01,
                          reassign_noise: bool = True,
                          use_original_embeddings: bool = False,
                          use_umap_embeddings: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings using HDBSCAN with improved parameters to reduce noise.
        Now defaults to using high-dimensional UMAP embeddings for better clustering.
        
        Updated tuning parameters (2025):
        - min_samples=5: Same as min_cluster_size for balance
        - cluster_selection_method='leaf': Fine-grained leaves for better cluster detection
        - cluster_selection_epsilon=0.0: Turn OFF auto-merging for more distinct clusters
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in a neighborhood for core points (default: 5, balanced with min_cluster_size)
            metric: Distance metric for clustering (default: 'euclidean')
            cluster_selection_method: How to select clusters from hierarchy (default: 'leaf' for fine-grained)
            cluster_selection_epsilon: Merge leaflets closer than this fraction of dist. range (default: 0.0 to disable merging)
            reassign_noise: Whether to use approximate_predict to reassign noise points (default: True)
            use_original_embeddings: If True, cluster on original high-D embeddings (overrides use_umap_embeddings)
            use_umap_embeddings: If True, cluster on UMAP-reduced embeddings (default: True, recommended)
            
        Returns:
            Tuple of (cluster_labels, outlier_scores)
        """
        # Choose which embeddings to use (priority order)
        if use_original_embeddings:
            if self.embeddings is None:
                raise ValueError("No original embeddings available. Call load_embeddings_from_db() first.")
            embeddings_to_cluster = self.embeddings
            logger.info("Using original high-dimensional embeddings for clustering")
            
        elif use_umap_embeddings:
            if self.embedding_umap is None:
                raise ValueError("No UMAP embeddings available. Call reduce_dimensionality() first.")
            embeddings_to_cluster = self.embedding_umap
            logger.info(f"Using {self.embedding_umap.shape[1]}D UMAP embeddings for clustering")
            
        else:
            # Fallback to 2D if specifically requested (not recommended)
            if self.embedding_2d is None:
                raise ValueError("No 2D embeddings available. Call reduce_dimensionality() first.")
            embeddings_to_cluster = self.embedding_2d
            logger.warning("Using 2D embeddings for clustering (not recommended - may exaggerate gaps)")
        
        # L2 normalize embeddings for cosine similarity equivalence with Euclidean distance
        embeddings_to_cluster = normalize(embeddings_to_cluster, norm='l2')
        logger.info(f"Applied L2 normalization to embeddings")
            
        logger.info(f"Clustering {embeddings_to_cluster.shape[0]} points (dimension: {embeddings_to_cluster.shape[1]}) using HDBSCAN with metric='{metric}', method='{cluster_selection_method}', epsilon={cluster_selection_epsilon}")
        
        # Initialize HDBSCAN with improved parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            prediction_data=True
        )
        
        # Fit and predict clusters
        self.cluster_labels = clusterer.fit_predict(embeddings_to_cluster)
        
        # Optional: reassign noise points using approximate_predict
        if reassign_noise:
            logger.info("Reassigning noise points using approximate_predict...")
            try:
                approximate_labels, _ = hdbscan.approximate_predict(clusterer, embeddings_to_cluster)
                
                # Count how many noise points were reassigned
                original_noise = list(self.cluster_labels).count(-1)
                new_noise = list(approximate_labels).count(-1)
                reassigned = original_noise - new_noise
                
                if reassigned > 0:
                    logger.info(f"Reassigned {reassigned} noise points to clusters")
                    self.cluster_labels = approximate_labels
                
            except Exception as e:
                logger.warning(f"Could not reassign noise points: {e}. Using original labels.")
        
        # Get outlier scores
        self.outlier_scores = clusterer.outlier_scores_
        
        # Add guard for cases where outlier_scores might not be available
        if self.outlier_scores is None:
            self.outlier_scores = np.zeros_like(self.cluster_labels, dtype=float)
        
        # Calculate statistics
        n_clusters = len(set(self.cluster_labels) - {-1})
        n_noise = list(self.cluster_labels).count(-1)
        
        logger.info(f"Found {n_clusters} clusters and {n_noise} noise/outlier points")
        
        return self.cluster_labels, self.outlier_scores
    
    def analyze_clusters(self) -> Dict[str, Any]:
        """
        Analyze cluster composition and identify outliers and small clusters.
        
        Returns:
            Dictionary with cluster analysis results
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering results available. Call cluster_embeddings() first.")
            
        # Count cluster sizes (excluding noise label -1)
        label_counts = Counter(label for label in self.cluster_labels if label != -1)
        n_noise = list(self.cluster_labels).count(-1)
        
        # Identify small clusters (fewer than 5 articles)
        small_clusters = {label: count for label, count in label_counts.items() if count < 5}
        
        # Get outlier information
        outlier_indices = np.where(self.cluster_labels == -1)[0]
        high_outlier_scores = np.where(self.outlier_scores > np.percentile(self.outlier_scores, 95))[0]
        
        analysis = {
            'n_clusters': len(label_counts),
            'n_noise': n_noise,
            'cluster_sizes': dict(label_counts),
            'small_clusters': small_clusters,
            'outlier_indices': outlier_indices.tolist(),
            'high_outlier_indices': high_outlier_scores.tolist(),
            'outlier_score_stats': {
                'mean': float(np.mean(self.outlier_scores)),
                'std': float(np.std(self.outlier_scores)),
                'min': float(np.min(self.outlier_scores)),
                'max': float(np.max(self.outlier_scores)),
                'percentiles': {
                    '95': float(np.percentile(self.outlier_scores, 95)),
                    '99': float(np.percentile(self.outlier_scores, 99))
                }
            }
        }
        
        # Print summary
        print(f"\n=== Cluster Analysis ===")
        print(f"Number of clusters: {analysis['n_clusters']}")
        print(f"Number of outliers (noise): {analysis['n_noise']}")
        print(f"Clusters with fewer than 5 articles: {len(small_clusters)}")
        
        for cluster_id in small_clusters:
            indices = [i for i, label in enumerate(self.cluster_labels) if label == cluster_id]
            cluster_titles = [self.titles[i] for i in indices]
            print(f"  - Cluster {cluster_id} (size={small_clusters[cluster_id]}): {cluster_titles[:3]}...")
            
        return analysis
    
    def visualize_clusters(self, 
                          save_html: bool = True,
                          filename: str = "cluster_visualization.html",
                          show_outliers: bool = True) -> go.Figure:
        """
        Create interactive visualization of clusters using Plotly.
        
        Args:
            save_html: Whether to save the plot as HTML
            filename: Filename for saved HTML
            show_outliers: Whether to highlight outliers
            
        Returns:
            Plotly figure object
        """
        if self.embedding_2d is None or self.cluster_labels is None:
            raise ValueError("Need both 2D embeddings and cluster labels. Make sure to run reduce_dimensionality() with create_2d_for_viz=True and cluster_embeddings().")
            
        # Prepare DataFrame for plotting
        df = pd.DataFrame({
            "x": self.embedding_2d[:, 0],
            "y": self.embedding_2d[:, 1],
            "title": self.titles,
            "id": self.ids,
            "cluster": self.cluster_labels,
            "outlier_score": self.outlier_scores
        })
        
        # Create cluster labels for coloring
        df["cluster_label"] = df["cluster"].apply(
            lambda c: "Outlier" if c == -1 else f"Cluster {c}"
        )
        
        # Create the main scatter plot
        fig = px.scatter(
            df, 
            x="x", 
            y="y", 
            color="cluster_label",
            hover_name="title",
            hover_data={
                "id": True, 
                "outlier_score": ":.3f",
                "x": False, 
                "y": False, 
                "cluster_label": False
            },
            title="UMAP Projection of Articles (2D Visualization, Clustered in 15D)"
        )
        
        # Highlight high outliers if requested
        if show_outliers:
            high_outlier_mask = df["outlier_score"] > np.percentile(df["outlier_score"], 95)
            high_outliers = df[high_outlier_mask]
            
            if not high_outliers.empty:
                fig.add_scatter(
                    x=high_outliers["x"],
                    y=high_outliers["y"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="red",
                        symbol="x",
                        line=dict(width=2, color="darkred")
                    ),
                    name="High Outliers",
                    hovertemplate="<b>%{hovertext}</b><br>High Outlier<br>Score: %{customdata:.3f}<extra></extra>",
                    hovertext=high_outliers["title"],
                    customdata=high_outliers["outlier_score"]
                )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            width=1000,
            height=700,
            showlegend=True
        )
        
        # Save if requested
        if save_html:
            fig.write_html(filename)
            logger.info(f"Visualization saved as {filename}")
            
        return fig
    
    def label_clusters_with_gpt4(self,
                                model: str = "gpt-4.1-mini",
                                max_titles_per_cluster: int = 5,
                                custom_prompt: Optional[str] = None) -> Dict[int, str]:
        """
        Use GPT-4 to generate labels for each cluster based on article titles.
        
        Args:
            model: OpenAI model to use
            max_titles_per_cluster: Maximum number of titles to send per cluster
            custom_prompt: Custom prompt template (optional)
            
        Returns:
            Dictionary mapping cluster IDs to generated labels
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering results available.")
            
        # Get unique cluster labels (excluding noise)
        unique_clusters = sorted([label for label in set(self.cluster_labels) if label != -1])
        cluster_names = {}
        
        # Default prompt template following 2025 best practices
        if custom_prompt is None:
            prompt_template = """I have a cluster of articles with the following titles:

{titles}

Based on these article titles, please provide a short, descriptive label (maximum 10 words) that captures the main theme or topic of this cluster.

Respond with only the label, nothing else."""
        else:
            prompt_template = custom_prompt
            
        logger.info(f"Labeling {len(unique_clusters)} clusters using {model}")
        
        for cluster_id in unique_clusters:
            try:
                # Get titles for this cluster
                cluster_indices = [i for i, label in enumerate(self.cluster_labels) if label == cluster_id]
                cluster_titles = [self.titles[i] for i in cluster_indices]
                
                # Sample titles if there are too many
                if len(cluster_titles) > max_titles_per_cluster:
                    cluster_titles = cluster_titles[:max_titles_per_cluster]
                
                # Format titles for prompt
                titles_text = "\n".join([f"- {title}" for title in cluster_titles])
                
                # Create the prompt
                prompt = prompt_template.format(titles=titles_text)
                
                # Call OpenAI API
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,  # Deterministic output
                    max_tokens=50   # Short labels
                )
                
                cluster_label = response.choices[0].message.content.strip()
                cluster_names[cluster_id] = cluster_label
                
                logger.info(f"Cluster {cluster_id}: {cluster_label}")
                
            except Exception as e:
                logger.error(f"Error labeling cluster {cluster_id}: {e}")
                cluster_names[cluster_id] = f"Cluster {cluster_id} (labeling failed)"
                
        return cluster_names
    
    def setup_bertopic_with_openai(self,
                                  model: str = "gpt-4.1-mini",
                                  embedding_model_name: str = "text-embedding-3-small",
                                  embedding_dims: int = 1536,
                                  use_multiple_representations: bool = True) -> BERTopic:
        """
        Set up BERTopic with OpenAI representation following 2025 best practices.
        
        Args:
            model: OpenAI model to use for topic labeling
            embedding_model_name: OpenAI embedding model to use
            embedding_dims: Dimensions for OpenAI embeddings (512 or 1536)
            use_multiple_representations: Whether to use multiple representation models
            
        Returns:
            Configured BERTopic model
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded.")
            
        # Create UMAP model
        umap_model = umap.UMAP(
            n_neighbors=30,  # Increased from 15 to 30 for denser manifold
            n_components=5, 
            min_dist=0.1, 
            metric='cosine',
            random_state=self.random_state
        )
        
        # Create HDBSCAN model
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=15, 
            metric='euclidean', 
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.06
        )
        
        # Create vectorizer model
        vectorizer_model = CountVectorizer(
            stop_words="english", 
            min_df=2, 
            ngram_range=(1, 2)
        )
        
        # Create OpenAI embedding model for consistency with database embeddings
        embedding_model = OpenAIEmbedder(
            model=embedding_model_name,
            dims=embedding_dims,
            client=self.openai_client
        )
        
        # Set up representation models
        representation_models = {}
        
        # OpenAI representation (2025 best practice prompt)
        openai_prompt = """
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
topic: <topic label>
"""
        
        openai_model = BERTopicOpenAI(
            client=self.openai_client,
            model=model,
            exponential_backoff=True,
            chat=True,
            prompt=openai_prompt,
            nr_docs=4,
            delay_in_seconds=1
        )
        representation_models["OpenAI"] = openai_model
        
        if use_multiple_representations:
            # KeyBERT-inspired representation using OpenAI embeddings
            keybert_model = KeyBERTInspired()
            representation_models["KeyBERT"] = keybert_model
            
            # MMR representation for diversity using OpenAI embeddings
            mmr_model = MaximalMarginalRelevance(diversity=0.3)
            representation_models["MMR"] = mmr_model
        
        # Create BERTopic model
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_models,
            top_n_words=10,
            verbose=True
        )
        
        return self.topic_model
    
    def run_bertopic_analysis(self, use_precomputed_embeddings: bool = True) -> Tuple[List[int], np.ndarray]:
        """
        Run complete BERTopic analysis.
        
        Args:
            use_precomputed_embeddings: Whether to use precomputed embeddings
            
        Returns:
            Tuple of (topics, probabilities)
        """
        if self.topic_model is None:
            raise ValueError("BERTopic model not set up. Call setup_bertopic_with_openai() first.")
            
        logger.info("Running BERTopic analysis...")
        
        if use_precomputed_embeddings and self.embeddings is not None:
            topics, probs = self.topic_model.fit_transform(self.titles, embeddings=self.embeddings)
        else:
            topics, probs = self.topic_model.fit_transform(self.titles)
            
        logger.info("BERTopic analysis complete!")
        return topics, probs
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information from BERTopic model."""
        if self.topic_model is None:
            raise ValueError("BERTopic model not fitted.")
        return self.topic_model.get_topic_info()
    
    def save_results(self, output_dir: str = "analysis_results"):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cluster analysis
        if self.cluster_labels is not None:
            analysis = self.analyze_clusters()
            
            # Save cluster assignments
            cluster_data = {
                'id': self.ids,
                'title': self.titles,
                'cluster': self.cluster_labels,
                'outlier_score': self.outlier_scores
            }
            
            # Only add UMAP coordinates if they exist
            if self.embedding_2d is not None:
                cluster_data['x_2d'] = self.embedding_2d[:, 0]
                cluster_data['y_2d'] = self.embedding_2d[:, 1]
            
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df.to_csv(f"{output_dir}/cluster_assignments.csv", index=False)
            
            # Save analysis summary
            with open(f"{output_dir}/cluster_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2)
                
        # Save BERTopic results if available
        if self.topic_model is not None:
            topic_info = self.get_topic_info()
            topic_info.to_csv(f"{output_dir}/bertopic_topics.csv", index=False)
            
        logger.info(f"Results saved to {output_dir}/")


def main():
    """
    Example usage of the EmbeddingAnalyzer class.
    """
    print("=== Article Embedding Clustering and Visualization ===\n")
    
    # Initialize analyzer
    analyzer = EmbeddingAnalyzer()
    
    try:
        # 1. Load embeddings from database
        print("1. Loading embeddings from PostgreSQL...")
        analyzer.load_embeddings_from_db(limit=1000)  # Limit for demo
        
        # 2. Reduce dimensionality with UMAP
        print("\n2. Reducing dimensionality with UMAP...")
        analyzer.reduce_dimensionality(n_components=15)  # 15D for clustering, 2D for visualization
        
        # 3. Cluster with HDBSCAN (using 15D UMAP embeddings)
        print("\n3. Clustering with HDBSCAN (using 15D UMAP embeddings)...")
        analyzer.cluster_embeddings(
            min_cluster_size=5, 
            metric='euclidean',  # Using Euclidean distance on L2-normalized embeddings (equivalent to cosine similarity)
            use_umap_embeddings=True  # Use 15D UMAP embeddings for clustering
        )
        
        # 4. Analyze clusters and identify outliers
        print("\n4. Analyzing clusters...")
        analysis = analyzer.analyze_clusters()
        
        # 5. Visualize clusters
        print("\n5. Creating visualization...")
        fig = analyzer.visualize_clusters()
        
        # 6. Label clusters with GPT-4
        print("\n6. Labeling clusters with GPT-4...")
        cluster_labels = analyzer.label_clusters_with_gpt4()
        
        # 7. Optional: Run BERTopic analysis
        print("\n7. Setting up BERTopic analysis...")
        analyzer.setup_bertopic_with_openai(
            model="gpt-4.1-mini",  
            embedding_model_name="text-embedding-3-small",
            embedding_dims=1536
        )
        topics, probs = analyzer.run_bertopic_analysis()
        
        # 8. Display results
        print("\n=== Results ===")
        print(f"Found {analysis['n_clusters']} clusters")
        print(f"Identified {analysis['n_noise']} outliers")
        
        print("\nCluster Labels:")
        for cluster_id, label in cluster_labels.items():
            size = analysis['cluster_sizes'][cluster_id]
            print(f"  Cluster {cluster_id} ({size} articles): {label}")
            
        # 9. Save results
        print("\n8. Saving results...")
        analyzer.save_results()
        
        print("\nAnalysis complete! Check 'analysis_results/' for detailed outputs.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
        print("\nMake sure to set up your environment variables:")
        print("- OPENAI_API_KEY: Your OpenAI API key (optional if skip_openai=True)")
        print("- AI_SAFETY_FEED_DB_URL: PostgreSQL connection string")


if __name__ == "__main__":
    main()
