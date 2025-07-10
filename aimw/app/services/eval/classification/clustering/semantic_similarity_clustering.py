import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import combinations
from scipy.stats import entropy


class SemanticSimilarityClusteringEvaluator:
    def __init__(
        self,
        text_model_name="all-MiniLM-L6-v2",
        prediction_model_name="all-MiniLM-L6-v2",
    ):
        """
        Initialize with sentence transformer models.

        Args:
            text_model_name: Model for document text embeddings
            prediction_model_name: Model for prediction/subtopic embeddings
        """
        self.text_embedding_model = SentenceTransformer(text_model_name)
        self.prediction_embedding_model = SentenceTransformer(prediction_model_name)
        self.text_embeddings = None
        self.clusters = None
        self.prediction_embeddings_cache = {}

    def load_data(self, contexts_file, predictions_file):
        """Load the JSON files and merge them."""
        with open(contexts_file, "r") as f:
            contexts = json.load(f)

        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        # Create mapping from doc_id to predictions
        pred_dict = {pred["doc_id"]: pred["classifiction"] for pred in predictions}

        # Merge contexts with predictions
        self.data = []
        for doc in contexts:
            doc_id = doc["doc_id"]
            if doc_id in pred_dict:
                self.data.append(
                    {
                        "doc_id": doc_id,
                        "text": doc["text"],
                        "predictions": pred_dict[doc_id],
                    }
                )

        print(f"Loaded {len(self.data)} documents with predictions")

        # Extract all unique predictions for embedding
        all_predictions = set()
        for doc in self.data:
            all_predictions.update(doc["predictions"])

        self.unique_predictions = list(all_predictions)
        print(
            f"Found {len(self.unique_predictions)} unique predictions: {self.unique_predictions}"
        )

        return self.data

    def load_merged_data(self, merged_file):
        """
        Load data from a single file containing merged context and prediction data.
        
        Expected format:
        {
            "docid": 22856,
            "doc": "document text content...",
            "perpectives": {
                "cir3_classifier_subtopics": ["topic1", "topic2", ...]
            }
        }
        
        Args:
            merged_file (str): Path to the JSON file containing merged data
            
        Returns:
            list: Processed data in the same format as load_data method
        """
        with open(merged_file, "r") as f:
            merged_data = json.load(f)

        # Process merged data to match the expected format
        self.data = []
        for item in merged_data:
            # Extract required fields
            doc_id = item.get("docid")
            text = item.get("doc")
            predictions = item.get("perpectives", {}).get("cir3_classifier_subtopics", [])
            
            if doc_id is not None and text is not None and predictions:
                self.data.append(
                    {
                        "doc_id": doc_id,
                        "text": text,
                        "predictions": predictions,
                    }
                )

        print(f"Loaded {len(self.data)} documents with predictions from merged file")

        # Extract all unique predictions for embedding
        all_predictions = set()
        for doc in self.data:
            all_predictions.update(doc["predictions"])

        self.unique_predictions = list(all_predictions)
        print(
            f"Found {len(self.unique_predictions)} unique predictions: {self.unique_predictions}"
        )

        return self.data

    def generate_text_embeddings(self):
        """Generate embeddings for document texts."""
        texts = [doc["text"] for doc in self.data]
        print("Generating text embeddings...")
        self.text_embeddings = self.text_embedding_model.encode(
            texts, show_progress_bar=True
        )
        print(f"Generated text embeddings with shape: {self.text_embeddings.shape}")
        return self.text_embeddings

    def generate_prediction_embeddings(self):
        """Generate embeddings for all unique predictions."""
        print("Generating prediction embeddings...")
        prediction_embeddings = self.prediction_embedding_model.encode(
            self.unique_predictions, show_progress_bar=True
        )

        # Create cache for quick lookup
        for i, pred in enumerate(self.unique_predictions):
            self.prediction_embeddings_cache[pred] = prediction_embeddings[i]

        print(
            f"Generated embeddings for {len(self.unique_predictions)} unique predictions"
        )
        return prediction_embeddings

    def semantic_prediction_similarity(
        self, predictions1, predictions2, method="max_alignment"
    ):
        """
        Calculate semantic similarity between two sets of predictions.

        Args:
            predictions1, predictions2: Lists of prediction strings
            method: 'max_alignment', 'average', 'hausdorff', or 'set_embedding'

        Returns:
            Similarity score between 0 and 1
        """
        if not predictions1 or not predictions2:
            return 0.0

        # Get embeddings for each prediction
        emb1 = np.array(
            [self.prediction_embeddings_cache[pred] for pred in predictions1]
        )
        emb2 = np.array(
            [self.prediction_embeddings_cache[pred] for pred in predictions2]
        )

        if method == "max_alignment":
            # For each prediction in set1, find max similarity in set2, then average
            similarities = []
            for e1 in emb1:
                max_sim = np.max(cosine_similarity([e1], emb2))
                similarities.append(max_sim)
            for e2 in emb2:
                max_sim = np.max(cosine_similarity([e2], emb1))
                similarities.append(max_sim)
            return np.mean(similarities)

        elif method == "average":
            # Average similarity between all pairs
            similarities = []
            for e1 in emb1:
                for e2 in emb2:
                    sim = cosine_similarity([e1], [e2])[0][0]
                    similarities.append(sim)
            return np.mean(similarities)

        elif method == "hausdorff":
            # Modified Hausdorff distance (symmetric)
            # Max of: max(min distances from set1 to set2), max(min distances from set2 to set1)
            distances1to2 = cdist(emb1, emb2, metric="cosine")
            distances2to1 = cdist(emb2, emb1, metric="cosine")

            # Convert distance to similarity and take min for each point
            max_sim1 = np.mean(1 - np.min(distances1to2, axis=1))
            max_sim2 = np.mean(1 - np.min(distances2to1, axis=1))

            return (max_sim1 + max_sim2) / 2

        elif method == "set_embedding":
            # Create set-level embeddings by averaging, then compare
            set_emb1 = np.mean(emb1, axis=0)
            set_emb2 = np.mean(emb2, axis=0)
            return cosine_similarity([set_emb1], [set_emb2])[0][0]

        else:
            raise ValueError(f"Unknown method: {method}")

    def find_optimal_clusters(self, max_clusters=20, method="kmeans"):
        """
        Find optimal number of clusters using silhouette analysis.
        Only used for KMeans clustering.
        """
        if self.text_embeddings is None:
            raise ValueError("Generate text embeddings first")

        if method != "kmeans":
            raise ValueError("find_optimal_clusters is only supported for KMeans clustering")

        scores = []
        cluster_range = range(2, min(max_clusters + 1, len(self.data)))

        for n_clusters in cluster_range:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(self.text_embeddings)

            silhouette_avg = silhouette_score(self.text_embeddings, cluster_labels)
            scores.append(silhouette_avg)
            print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")

        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, scores, "bo-")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis for Optimal Clusters")
        plt.grid(True)
        plt.show()

        optimal_clusters = cluster_range[np.argmax(scores)]
        print(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters

    def perform_clustering(self, n_clusters=None, method="kmeans", dbscan_eps=0.5, dbscan_min_samples=5):
        """
        Perform clustering on text embeddings.
        
        Args:
            n_clusters (int, optional): Number of clusters for KMeans. If None, will be determined automatically.
            method (str): Clustering method to use ('kmeans' or 'dbscan')
            dbscan_eps (float): The maximum distance between two samples for DBSCAN
            dbscan_min_samples (int): The number of samples in a neighborhood for a point to be considered a core point
        
        Returns:
            numpy.ndarray: Cluster labels for each document
        """
        if self.text_embeddings is None:
            raise ValueError("Generate text embeddings first")

        if method == "kmeans":
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters(method="kmeans")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            self.clusters = clusterer.fit_predict(self.text_embeddings)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            self.clusters = clusterer.fit_predict(self.text_embeddings)
            # Print DBSCAN specific information
            n_clusters_ = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
            n_noise_ = list(self.clusters).count(-1)
            print(f'Estimated number of DBSCAN clusters: {n_clusters_}')
            print(f'Estimated number of noise points: {n_noise_}')
        else:
            raise ValueError("method must be either 'kmeans' or 'dbscan'")

        # Add cluster labels to data
        for i, doc in enumerate(self.data):
            doc["cluster"] = int(self.clusters[i])

        # Calculate clustering quality metrics if possible
        unique_clusters = set(self.clusters)
        if len(unique_clusters) > 1 and -1 not in self.clusters:  # Skip if there's only one cluster or if there are noise points
            try:
                sil_score = silhouette_score(self.text_embeddings, self.clusters)
                print(f"Silhouette Score: {sil_score:.3f}")
            except ValueError as e:
                print(f"Could not calculate silhouette score: {e}")
        elif len(unique_clusters) == 1:
            print("Warning: Only 1 cluster found - silhouette score cannot be calculated")
        elif -1 in self.clusters:
            print("Warning: DBSCAN found noise points - silhouette score cannot be calculated")

        print(f"Clustering complete. Found {len(unique_clusters)} clusters")
        return self.clusters

    def analyze_semantic_prediction_consistency(
        self, similarity_methods=["max_alignment", "set_embedding"]
    ):
        """Analyze consistency using semantic similarity."""
        if self.clusters is None:
            raise ValueError("Perform clustering first")

        cluster_analysis = defaultdict(list)

        # Group documents by cluster
        for doc in self.data:
            cluster_id = doc["cluster"]
            cluster_analysis[cluster_id].append(
                {
                    "doc_id": doc["doc_id"],
                    "predictions": doc["predictions"],
                    "text_preview": doc["text"][:100] + "...",
                }
            )

        consistency_scores = {}

        for cluster_id, docs in cluster_analysis.items():
            if len(docs) < 2:
                continue

            cluster_scores = {}

            for method in similarity_methods:
                # Calculate semantic similarity for all pairs in cluster
                similarities = []
                all_predictions = [doc["predictions"] for doc in docs]

                for i in range(len(all_predictions)):
                    for j in range(i + 1, len(all_predictions)):
                        pred1, pred2 = all_predictions[i], all_predictions[j]
                        sem_sim = self.semantic_prediction_similarity(
                            pred1, pred2, method=method
                        )
                        similarities.append(sem_sim)

                avg_similarity = np.mean(similarities) if similarities else 0
                cluster_scores[f"{method}_similarity"] = avg_similarity

            # Also calculate syntactic Jaccard for comparison
            jaccard_similarities = []
            for i in range(len(all_predictions)):
                for j in range(i + 1, len(all_predictions)):
                    pred1, pred2 = set(all_predictions[i]), set(all_predictions[j])
                    intersection = len(pred1.intersection(pred2))
                    union = len(pred1.union(pred2))
                    jaccard = intersection / union if union > 0 else 0
                    jaccard_similarities.append(jaccard)

            cluster_scores["jaccard_similarity"] = (
                np.mean(jaccard_similarities) if jaccard_similarities else 0
            )

            consistency_scores[cluster_id] = {
                **cluster_scores,
                "num_docs": len(docs),
                "unique_predictions": list(set().union(*all_predictions)),
                "docs": docs,
                "prediction_diversity": len(set().union(*all_predictions)),
            }

        return consistency_scores

    def compare_similarity_methods(self):
        """Compare different similarity methods on sample prediction pairs."""
        print("\n=== Comparing Similarity Methods ===")

        # Create some example prediction pairs to demonstrate differences
        sample_pairs = [
            (["mortgage", "home loan"], ["mortgage", "refinancing"]),
            (["pension", "retirement"], ["401k", "retirement planning"]),
            (["life insurance"], ["term insurance", "whole life"]),
            (["investment", "portfolio"], ["stocks", "bonds"]),
            (["mortgage"], ["investment"]),  # Should be low similarity
        ]

        methods = ["max_alignment", "average", "hausdorff", "set_embedding"]

        results = []
        for pred1, pred2 in sample_pairs:
            row = {"pred1": pred1, "pred2": pred2}

            # Jaccard for comparison
            set1, set2 = set(pred1), set(pred2)
            jaccard = (
                len(set1.intersection(set2)) / len(set1.union(set2))
                if len(set1.union(set2)) > 0
                else 0
            )
            row["jaccard"] = jaccard

            # Semantic methods
            for method in methods:
                try:
                    sim = self.semantic_prediction_similarity(
                        pred1, pred2, method=method
                    )
                    row[method] = sim
                except:
                    row[method] = 0.0

            results.append(row)

        # Display results
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format="%.3f"))

        return df

    def identify_inconsistent_clusters(self, threshold=0.5, method="max_alignment"):
        """Identify clusters with low semantic prediction consistency."""
        consistency_scores = self.analyze_semantic_prediction_consistency()

        similarity_key = f"{method}_similarity"
        inconsistent_clusters = []

        for cluster_id, scores in consistency_scores.items():
            if scores[similarity_key] < threshold:
                inconsistent_clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "semantic_consistency": scores[similarity_key],
                        "jaccard_consistency": scores["jaccard_similarity"],
                        "improvement": scores[similarity_key]
                        - scores["jaccard_similarity"],
                        "num_docs": scores["num_docs"],
                        "prediction_diversity": scores["prediction_diversity"],
                        "sample_docs": scores["docs"][:3],
                    }
                )

        # Sort by semantic consistency score (lowest first)
        inconsistent_clusters.sort(key=lambda x: x["semantic_consistency"])

        return inconsistent_clusters

    def analyze_prediction_distribution(self):
        """Analyze the distribution and diversity of predictions across all documents."""
        # Count subtopic frequencies
        subtopic_counts = Counter()
        for doc in self.data:
            subtopic_counts.update(doc["predictions"])

        # Compute entropy
        total_docs = len(self.data)
        subtopic_probs = np.array(
            [count / total_docs for count in subtopic_counts.values()]
        )
        subtopic_entropy = entropy(subtopic_probs, base=2)

        # Calculate additional metrics
        max_entropy = np.log2(len(subtopic_counts))  # Maximum possible entropy
        normalized_entropy = subtopic_entropy / max_entropy if max_entropy > 0 else 0

        distribution_metrics = {
            "num_unique_predictions": len(subtopic_counts),
            "prediction_entropy": float(
                subtopic_entropy
            ),  # Convert to float for JSON serialization
            "normalized_entropy": float(normalized_entropy),
            "prediction_frequencies": dict(subtopic_counts),
            "most_common_predictions": dict(subtopic_counts.most_common(5)),
            "least_common_predictions": dict(subtopic_counts.most_common()[:-6:-1]),
        }

        return distribution_metrics

    def calculate_overall_consistency(self, similarity_methods=["max_alignment", "set_embedding"]):
        """Calculate consistency across all documents, not just within clusters."""
        overall_scores = {}
        
        for method in similarity_methods:
            # Calculate similarities between all document pairs
            similarities = []
            for i in range(len(self.data)):
                for j in range(i + 1, len(self.data)):
                    pred1 = self.data[i]["predictions"]
                    pred2 = self.data[j]["predictions"]
                    sem_sim = self.semantic_prediction_similarity(pred1, pred2, method=method)
                    similarities.append(sem_sim)
            
            # Calculate overall statistics
            similarities = np.array(similarities)
            overall_scores[f"{method}_similarity"] = {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities)),
                "median": float(np.median(similarities)),
                "total_comparisons": len(similarities)
            }
            
        # Calculate Jaccard similarity for comparison
        jaccard_similarities = []
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                pred1 = set(self.data[i]["predictions"])
                pred2 = set(self.data[j]["predictions"])
                intersection = len(pred1.intersection(pred2))
                union = len(pred1.union(pred2))
                jaccard = intersection / union if union > 0 else 0
                jaccard_similarities.append(jaccard)
                
        jaccard_similarities = np.array(jaccard_similarities)
        overall_scores["jaccard_similarity"] = {
            "mean": float(np.mean(jaccard_similarities)),
            "std": float(np.std(jaccard_similarities)),
            "min": float(np.min(jaccard_similarities)),
            "max": float(np.max(jaccard_similarities)),
            "median": float(np.median(jaccard_similarities)),
            "total_comparisons": len(jaccard_similarities)
        }
        
        return overall_scores

    def generate_detailed_evaluation_report(self, output_file="semantic_evaluation_report.json"):
        """Generate comprehensive evaluation report with semantic analysis."""
        # Analyze with multiple methods
        consistency_scores = self.analyze_semantic_prediction_consistency()
        inconsistent_clusters = self.identify_inconsistent_clusters()

        # Method comparison
        method_comparison = self.compare_similarity_methods()

        # Add distribution analysis
        distribution_metrics = self.analyze_prediction_distribution()

        # Calculate overall document-level consistency
        overall_consistency = self.calculate_overall_consistency()

        # Overall statistics for each method
        methods = ["max_alignment_similarity", "set_embedding_similarity", "jaccard_similarity"]
        overall_stats = {}

        for method in methods:
            similarities = [scores[method] for scores in consistency_scores.values() if method in scores]
            if similarities:
                overall_stats[method] = {
                    "mean": np.mean(similarities),
                    "std": np.std(similarities),
                    "min": np.min(similarities),
                    "max": np.max(similarities),
                    "median": np.median(similarities),
                }

        report = {
            "overall_metrics": {
                "total_documents": len(self.data),
                "total_clusters": len(set(self.clusters)),
                "unique_predictions": len(self.unique_predictions),
                "prediction_list": self.unique_predictions,
                "method_statistics": overall_stats,
                "overall_document_consistency": overall_consistency,
                "prediction_distribution": distribution_metrics,
            },
            "cluster_details": consistency_scores,
            "inconsistent_clusters": inconsistent_clusters,
            "method_comparison_examples": method_comparison.to_dict("records"),
            "recommendations": self._generate_semantic_recommendations(inconsistent_clusters, overall_stats),
        }

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Detailed evaluation report saved to {output_file}")
        return report

    def _generate_semantic_recommendations(self, inconsistent_clusters, overall_stats):
        """Generate recommendations based on semantic analysis."""
        recommendations = []

        if inconsistent_clusters:
            recommendations.append(
                f"Found {len(inconsistent_clusters)} clusters with low semantic consistency. "
                "These represent areas where your model predictions are inconsistent for similar content."
            )

        # Compare semantic vs syntactic performance
        if (
            "max_alignment_similarity" in overall_stats
            and "jaccard_similarity" in overall_stats
        ):
            semantic_avg = overall_stats["max_alignment_similarity"]["mean"]
            jaccard_avg = overall_stats["jaccard_similarity"]["mean"]
            improvement = semantic_avg - jaccard_avg

            if improvement > 0.1:
                recommendations.append(
                    f"Semantic similarity shows {improvement:.3f} higher consistency than syntactic Jaccard. "
                    "Your model may be using semantically related but syntactically different terms."
                )
            elif improvement < -0.1:
                recommendations.append(
                    f"Semantic similarity shows lower consistency than Jaccard. "
                    "This suggests your model might be using inconsistent terminology for similar concepts."
                )

        # Prediction diversity analysis
        high_diversity_clusters = [
            c for c in inconsistent_clusters if c["prediction_diversity"] > 5
        ]
        if high_diversity_clusters:
            recommendations.append(
                f"Found {len(high_diversity_clusters)} clusters with high prediction diversity. "
                "Consider if your model is over-predicting or needs more focused few-shot examples."
            )

        return recommendations

    def visualize_prediction_similarity_matrix(self):
        """Visualize similarity matrix between all unique predictions."""
        if not hasattr(self, "prediction_embeddings_cache"):
            raise ValueError("Generate prediction embeddings first")

        # Create similarity matrix
        predictions = self.unique_predictions
        n_preds = len(predictions)
        similarity_matrix = np.zeros((n_preds, n_preds))

        for i in range(n_preds):
            for j in range(n_preds):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    emb_i = self.prediction_embeddings_cache[predictions[i]]
                    emb_j = self.prediction_embeddings_cache[predictions[j]]
                    similarity_matrix[i][j] = cosine_similarity([emb_i], [emb_j])[0][0]

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(similarity_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(label="Cosine Similarity")
        plt.title("Semantic Similarity Matrix of Predictions")
        plt.xlabel("Predictions")
        plt.ylabel("Predictions")

        # Add labels
        plt.xticks(range(n_preds), predictions, rotation=45, ha="right")
        plt.yticks(range(n_preds), predictions)

        # Add similarity values as text
        for i in range(n_preds):
            for j in range(n_preds):
                plt.text(
                    j,
                    i,
                    f"{similarity_matrix[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if similarity_matrix[i][j] < 0.5 else "black",
                )

        plt.tight_layout()
        plt.show()

        return similarity_matrix

    def diagnose_clustering_issues(self):
        """
        Diagnose why clustering might be failing and suggest solutions.
        """
        if self.text_embeddings is None:
            raise ValueError("Generate text embeddings first")
        
        print("=== CLUSTERING DIAGNOSIS ===")
        
        # 1. Check embedding statistics
        print(f"\n1. Embedding Statistics:")
        print(f"   - Number of documents: {len(self.data)}")
        print(f"   - Embedding dimensions: {self.text_embeddings.shape[1]}")
        print(f"   - Embedding range: [{self.text_embeddings.min():.4f}, {self.text_embeddings.max():.4f}]")
        print(f"   - Embedding mean: {self.text_embeddings.mean():.4f}")
        print(f"   - Embedding std: {self.text_embeddings.std():.4f}")
        
        # 2. Calculate pairwise distances
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(self.text_embeddings)
        
        print(f"\n2. Distance Statistics:")
        print(f"   - Min distance: {distances[distances > 0].min():.4f}")
        print(f"   - Max distance: {distances.max():.4f}")
        print(f"   - Mean distance: {distances[distances > 0].mean():.4f}")
        print(f"   - Median distance: {np.median(distances[distances > 0]):.4f}")
        
        # 3. Check for duplicate or very similar documents
        very_similar_pairs = np.where(distances < 0.01)
        very_similar_pairs = [(i, j) for i, j in zip(very_similar_pairs[0], very_similar_pairs[1]) if i < j]
        
        print(f"\n3. Similarity Analysis:")
        print(f"   - Documents with distance < 0.01: {len(very_similar_pairs)} pairs")
        print(f"   - Documents with distance < 0.05: {len(np.where(distances < 0.05)[0]) // 2} pairs")
        print(f"   - Documents with distance < 0.1: {len(np.where(distances < 0.1)[0]) // 2} pairs")
        
        # 4. Analyze prediction diversity
        all_predictions = []
        for doc in self.data:
            all_predictions.extend(doc["predictions"])
        
        unique_predictions = set(all_predictions)
        prediction_counts = Counter(all_predictions)
        
        print(f"\n4. Prediction Analysis:")
        print(f"   - Total predictions: {len(all_predictions)}")
        print(f"   - Unique predictions: {len(unique_predictions)}")
        print(f"   - Average predictions per doc: {len(all_predictions) / len(self.data):.2f}")
        print(f"   - Most common prediction: '{prediction_counts.most_common(1)[0][0]}' ({prediction_counts.most_common(1)[0][1]} times)")
        
        # 5. Suggest solutions
        print(f"\n5. RECOMMENDATIONS:")
        
        if len(very_similar_pairs) > len(self.data) * 0.1:
            print("   ⚠️  Many documents are very similar - consider:")
            print("      - Using different embedding model")
            print("      - Preprocessing documents to increase diversity")
            print("      - Using prediction-based clustering instead")
        
        if distances.max() < 0.5:
            print("   ⚠️  All documents are very close together - consider:")
            print("      - Using smaller eps values for DBSCAN (try 0.1, 0.05, 0.01)")
            print("      - Using prediction embeddings instead of document embeddings")
            print("      - Normalizing embeddings")
        
        if len(unique_predictions) > len(self.data) * 2:
            print("   ⚠️  Very diverse predictions - consider:")
            print("      - Clustering on prediction embeddings instead")
            print("      - Using hierarchical clustering")
            print("      - Focusing on semantic consistency rather than clustering")
        
        print(f"\n6. SUGGESTED NEXT STEPS:")
        print("   - Try clustering on prediction embeddings: evaluator.cluster_on_predictions()")
        print("   - Try different DBSCAN parameters: evaluator.perform_clustering(method='dbscan', dbscan_eps=0.1, dbscan_min_samples=2)")
        print("   - Try hierarchical clustering: evaluator.hierarchical_clustering()")
        print("   - Focus on semantic consistency analysis instead of clustering")

    def cluster_on_predictions(self, method="kmeans", n_clusters=None, dbscan_eps=0.3, dbscan_min_samples=2):
        """
        Cluster documents based on their prediction embeddings instead of document embeddings.
        This might work better when documents are similar but predictions are diverse.
        """
        if not hasattr(self, 'prediction_embeddings_cache'):
            raise ValueError("Generate prediction embeddings first")
        
        print("=== CLUSTERING ON PREDICTIONS ===")
        
        # Create document-level prediction embeddings by averaging all prediction embeddings for each document
        doc_prediction_embeddings = []
        for doc in self.data:
            pred_embeddings = [self.prediction_embeddings_cache[pred] for pred in doc["predictions"]]
            doc_pred_emb = np.mean(pred_embeddings, axis=0)
            doc_prediction_embeddings.append(doc_pred_emb)
        
        doc_prediction_embeddings = np.array(doc_prediction_embeddings)
        
        print(f"Created prediction-based embeddings with shape: {doc_prediction_embeddings.shape}")
        
        # Perform clustering on prediction embeddings
        if method == "kmeans":
            if n_clusters is None:
                # Try to find optimal clusters for prediction embeddings
                scores = []
                cluster_range = range(2, min(10, len(self.data)))
                
                for n_clusters in cluster_range:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = clusterer.fit_predict(doc_prediction_embeddings)
                    try:
                        silhouette_avg = silhouette_score(doc_prediction_embeddings, cluster_labels)
                        scores.append(silhouette_avg)
                        print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")
                    except ValueError:
                        scores.append(-1)
                        print(f"Clusters: {n_clusters}, Silhouette Score: N/A (insufficient clusters)")
                
                if max(scores) > -1:
                    n_clusters = cluster_range[np.argmax(scores)]
                    print(f"Optimal number of clusters: {n_clusters}")
                else:
                    n_clusters = 2  # fallback
                    print("Using fallback: 2 clusters")
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            self.clusters = clusterer.fit_predict(doc_prediction_embeddings)
            
        elif method == "dbscan":
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            self.clusters = clusterer.fit_predict(doc_prediction_embeddings)
            
            n_clusters_ = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
            n_noise_ = list(self.clusters).count(-1)
            print(f'Estimated number of DBSCAN clusters: {n_clusters_}')
            print(f'Estimated number of noise points: {n_noise_}')
        
        # Add cluster labels to data
        for i, doc in enumerate(self.data):
            doc["cluster"] = int(self.clusters[i])
        
        # Calculate clustering quality metrics if possible
        unique_clusters = set(self.clusters)
        if len(unique_clusters) > 1 and -1 not in self.clusters:
            try:
                sil_score = silhouette_score(doc_prediction_embeddings, self.clusters)
                print(f"Silhouette Score: {sil_score:.3f}")
            except ValueError as e:
                print(f"Could not calculate silhouette score: {e}")
        elif len(unique_clusters) == 1:
            print("Warning: Only 1 cluster found - silhouette score cannot be calculated")
        elif -1 in self.clusters:
            print("Warning: DBSCAN found noise points - silhouette score cannot be calculated")
        
        print(f"Clustering complete. Found {len(unique_clusters)} clusters")
        return self.clusters

    def hierarchical_clustering(self, n_clusters=None, distance_threshold=None):
        """
        Perform hierarchical clustering which might work better for this type of data.
        """
        from sklearn.cluster import AgglomerativeClustering
        
        if self.text_embeddings is None:
            raise ValueError("Generate text embeddings first")
        
        print("=== HIERARCHICAL CLUSTERING ===")
        
        if n_clusters is not None:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif distance_threshold is not None:
            clusterer = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
        else:
            # Try different distance thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            best_threshold = None
            best_score = -1
            
            for threshold in thresholds:
                clusterer = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None)
                labels = clusterer.fit_predict(self.text_embeddings)
                
                if len(set(labels)) > 1:
                    try:
                        score = silhouette_score(self.text_embeddings, labels)
                        print(f"Threshold {threshold}: {len(set(labels))} clusters, Silhouette: {score:.3f}")
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
                    except ValueError:
                        print(f"Threshold {threshold}: {len(set(labels))} clusters, Silhouette: N/A")
                else:
                    print(f"Threshold {threshold}: {len(set(labels))} clusters, Silhouette: N/A (insufficient clusters)")
            
            if best_threshold is not None:
                clusterer = AgglomerativeClustering(distance_threshold=best_threshold, n_clusters=None)
                print(f"Using best threshold: {best_threshold}")
            else:
                clusterer = AgglomerativeClustering(n_clusters=2)  # fallback
                print("Using fallback: 2 clusters")
        
        self.clusters = clusterer.fit_predict(self.text_embeddings)
        
        # Add cluster labels to data
        for i, doc in enumerate(self.data):
            doc["cluster"] = int(self.clusters[i])
        
        # Calculate clustering quality metrics if possible
        unique_clusters = set(self.clusters)
        if len(unique_clusters) > 1:
            try:
                sil_score = silhouette_score(self.text_embeddings, self.clusters)
                print(f"Silhouette Score: {sil_score:.3f}")
            except ValueError as e:
                print(f"Could not calculate silhouette score: {e}")
        else:
            print("Warning: Only 1 cluster found - silhouette score cannot be calculated")
        
        print(f"Hierarchical clustering complete. Found {len(unique_clusters)} clusters")
        return self.clusters

    def find_optimal_clusters_for_predictions(self, max_clusters=15, method="silhouette"):
        """
        Find optimal number of clusters for prediction-based clustering using different methods.
        
        Args:
            max_clusters (int): Maximum number of clusters to try
            method (str): Method to use ('silhouette', 'elbow', 'gap_statistic')
        
        Returns:
            int: Optimal number of clusters
        """
        if not hasattr(self, 'prediction_embeddings_cache'):
            raise ValueError("Generate prediction embeddings first")
        
        # Create document-level prediction embeddings
        doc_prediction_embeddings = []
        for doc in self.data:
            pred_embeddings = [self.prediction_embeddings_cache[pred] for pred in doc["predictions"]]
            doc_pred_emb = np.mean(pred_embeddings, axis=0)
            doc_prediction_embeddings.append(doc_pred_emb)
        
        doc_prediction_embeddings = np.array(doc_prediction_embeddings)
        
        cluster_range = range(2, min(max_clusters + 1, len(self.data)))
        
        if method == "silhouette":
            scores = []
            for n_clusters in cluster_range:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(doc_prediction_embeddings)
                try:
                    silhouette_avg = silhouette_score(doc_prediction_embeddings, cluster_labels)
                    scores.append(silhouette_avg)
                    print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")
                except ValueError:
                    scores.append(-1)
                    print(f"Clusters: {n_clusters}, Silhouette Score: N/A")
            
            if max(scores) > -1:
                optimal_clusters = cluster_range[np.argmax(scores)]
                print(f"Optimal number of clusters (silhouette): {optimal_clusters}")
                
                # Plot silhouette scores
                plt.figure(figsize=(10, 6))
                plt.plot(cluster_range, scores, "bo-")
                plt.xlabel("Number of Clusters")
                plt.ylabel("Silhouette Score")
                plt.title("Silhouette Analysis for Prediction-Based Clustering")
                plt.grid(True)
                plt.show()
                
                return optimal_clusters
            else:
                print("Could not determine optimal clusters using silhouette method")
                return 2
        
        elif method == "elbow":
            inertias = []
            for n_clusters in cluster_range:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                clusterer.fit(doc_prediction_embeddings)
                inertias.append(clusterer.inertia_)
                print(f"Clusters: {n_clusters}, Inertia: {clusterer.inertia_:.2f}")
            
            # Plot elbow curve
            plt.figure(figsize=(10, 6))
            plt.plot(cluster_range, inertias, "bo-")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Inertia")
            plt.title("Elbow Method for Prediction-Based Clustering")
            plt.grid(True)
            plt.show()
            
            # Find elbow point (where the rate of decrease slows down)
            # Simple method: find the point with maximum second derivative
            if len(inertias) > 2:
                second_derivatives = []
                for i in range(1, len(inertias) - 1):
                    second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                    second_derivatives.append(second_deriv)
                
                elbow_idx = np.argmax(second_derivatives) + 1
                optimal_clusters = cluster_range[elbow_idx]
                print(f"Optimal number of clusters (elbow): {optimal_clusters}")
                return optimal_clusters
            else:
                print("Not enough data points for elbow method")
                return 2
        
        elif method == "gap_statistic":
            from sklearn.metrics import silhouette_score
            from sklearn.utils import resample
            
            def compute_gap_statistic(data, k, n_references=10):
                """Compute gap statistic for a given k."""
                # Fit KMeans to original data
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data)
                original_inertia = kmeans.inertia_
                
                # Generate reference datasets
                reference_inertias = []
                for _ in range(n_references):
                    # Generate random data with same range as original
                    min_vals = data.min(axis=0)
                    max_vals = data.max(axis=0)
                    random_data = np.random.uniform(min_vals, max_vals, data.shape)
                    
                    kmeans_ref = KMeans(n_clusters=k, random_state=42)
                    kmeans_ref.fit(random_data)
                    reference_inertias.append(kmeans_ref.inertia_)
                
                # Compute gap statistic
                gap = np.mean(np.log(reference_inertias)) - np.log(original_inertia)
                return gap
            
            gaps = []
            for n_clusters in cluster_range:
                gap = compute_gap_statistic(doc_prediction_embeddings, n_clusters)
                gaps.append(gap)
                print(f"Clusters: {n_clusters}, Gap Statistic: {gap:.3f}")
            
            # Plot gap statistics
            plt.figure(figsize=(10, 6))
            plt.plot(cluster_range, gaps, "bo-")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Gap Statistic")
            plt.title("Gap Statistic for Prediction-Based Clustering")
            plt.grid(True)
            plt.show()
            
            optimal_clusters = cluster_range[np.argmax(gaps)]
            print(f"Optimal number of clusters (gap statistic): {optimal_clusters}")
            return optimal_clusters
        
        else:
            raise ValueError("method must be 'silhouette', 'elbow', or 'gap_statistic'")

    def cluster_on_predictions_with_threshold(self, method="kmeans", n_clusters=None, dbscan_eps=0.3, dbscan_min_samples=2, 
                                            silhouette_threshold=0.1, min_clusters=2, max_clusters=10):
        """
        Enhanced clustering on predictions with threshold-based validation.
        
        Args:
            method (str): Clustering method ('kmeans' or 'dbscan')
            n_clusters (int): Number of clusters for KMeans
            dbscan_eps (float): DBSCAN epsilon parameter
            dbscan_min_samples (int): DBSCAN min_samples parameter
            silhouette_threshold (float): Minimum silhouette score to accept clustering
            min_clusters (int): Minimum number of clusters to accept
            max_clusters (int): Maximum number of clusters to try
        """
        if not hasattr(self, 'prediction_embeddings_cache'):
            raise ValueError("Generate prediction embeddings first")
        
        print("=== ENHANCED CLUSTERING ON PREDICTIONS ===")
        
        # Create document-level prediction embeddings
        doc_prediction_embeddings = []
        for doc in self.data:
            pred_embeddings = [self.prediction_embeddings_cache[pred] for pred in doc["predictions"]]
            doc_pred_emb = np.mean(pred_embeddings, axis=0)
            doc_prediction_embeddings.append(doc_pred_emb)
        
        doc_prediction_embeddings = np.array(doc_prediction_embeddings)
        print(f"Created prediction-based embeddings with shape: {doc_prediction_embeddings.shape}")
        
        if method == "kmeans":
            if n_clusters is None:
                # Try different numbers of clusters and find the best one
                best_n_clusters = min_clusters
                best_silhouette = -1
                best_clusters = None
                
                for k in range(min_clusters, min(max_clusters + 1, len(self.data))):
                    clusterer = KMeans(n_clusters=k, random_state=42)
                    cluster_labels = clusterer.fit_predict(doc_prediction_embeddings)
                    
                    if len(set(cluster_labels)) > 1:
                        try:
                            sil_score = silhouette_score(doc_prediction_embeddings, cluster_labels)
                            print(f"Testing {k} clusters: Silhouette = {sil_score:.3f}")
                            
                            if sil_score > best_silhouette and sil_score >= silhouette_threshold:
                                best_silhouette = sil_score
                                best_n_clusters = k
                                best_clusters = cluster_labels
                        except ValueError:
                            print(f"Testing {k} clusters: Silhouette = N/A")
                    else:
                        print(f"Testing {k} clusters: Only 1 cluster found")
                
                if best_clusters is not None:
                    self.clusters = best_clusters
                    print(f"Selected {best_n_clusters} clusters with silhouette score {best_silhouette:.3f}")
                else:
                    print(f"No clustering met the threshold ({silhouette_threshold}). Using {min_clusters} clusters.")
                    clusterer = KMeans(n_clusters=min_clusters, random_state=42)
                    self.clusters = clusterer.fit_predict(doc_prediction_embeddings)
            else:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                self.clusters = clusterer.fit_predict(doc_prediction_embeddings)
        
        elif method == "dbscan":
            # Try different eps values for DBSCAN
            eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            best_eps = dbscan_eps
            best_n_clusters = 0
            best_clusters = None
            
            for eps in eps_values:
                clusterer = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
                cluster_labels = clusterer.fit_predict(doc_prediction_embeddings)
                n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise_ = list(cluster_labels).count(-1)
                
                print(f"Testing eps={eps}: {n_clusters_} clusters, {n_noise_} noise points")
                
                if n_clusters_ >= min_clusters and n_clusters_ <= max_clusters:
                    if n_clusters_ > best_n_clusters:
                        best_n_clusters = n_clusters_
                        best_eps = eps
                        best_clusters = cluster_labels
            
            if best_clusters is not None:
                self.clusters = best_clusters
                print(f"Selected eps={best_eps} with {best_n_clusters} clusters")
            else:
                print(f"No DBSCAN parameters found. Using default eps={dbscan_eps}")
                clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                self.clusters = clusterer.fit_predict(doc_prediction_embeddings)
        
        # Add cluster labels to data
        for i, doc in enumerate(self.data):
            doc["cluster"] = int(self.clusters[i])
        
        # Calculate final clustering quality metrics
        unique_clusters = set(self.clusters)
        if len(unique_clusters) > 1 and -1 not in self.clusters:
            try:
                sil_score = silhouette_score(doc_prediction_embeddings, self.clusters)
                print(f"Final Silhouette Score: {sil_score:.3f}")
            except ValueError as e:
                print(f"Could not calculate final silhouette score: {e}")
        elif len(unique_clusters) == 1:
            print("Warning: Only 1 cluster found")
        elif -1 in self.clusters:
            print("Warning: DBSCAN found noise points")
        
        print(f"Clustering complete. Found {len(unique_clusters)} clusters")
        return self.clusters