
"""
EduRetain Model Training Pipeline

Implements the hybrid machine learning approach:
1. Unsupervised Learning: K-Means clustering with PCA
2. Supervised Learning: Random Forest classifier per cluster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class EduRetainTrainer:
    """
    Hybrid ML model trainer for dropout prediction.
    
    Phase 1: Unsupervised clustering to discover learner types
    Phase 2: Supervised classification per cluster for dropout prediction
    """
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, random_state=random_state)  # 95% variance
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.rf_models = {}  # One Random Forest per cluster
        
        # Results
        self.cluster_profiles = {}
        self.performance_metrics = {}
        
    def prepare_features(self, df):
        """Extract and prepare features for modeling."""
        
        print("🔧 Preparing features...")
        
        # Features for clustering (behavioral patterns)
        clustering_features = [
            'video_completion_rate', 'rewatch_rate', 'avg_playback_speed',
            'video_pause_frequency', 'skip_rate',
            'quiz_avg_score', 'quiz_score_std', 'quiz_attempts_avg',
            'assignment_submission_rate', 'score_improvement',
            'forum_post_count', 'forum_reply_count', 'forum_upvotes_received',
            'login_frequency_per_week', 'avg_session_duration_min',
            'days_since_last_login', 'study_time_variance', 'deadline_adherence_rate',
            'percent_complete', 'weeks_enrolled',
            'engagement_score', 'performance_score'
        ]
        
        X = df[clustering_features].copy()
        y = df['dropped_out'].copy()
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]:,}")
        print(f"   Dropout rate: {y.mean():.1%}")
        
        return X, y, clustering_features
    
    def phase1_unsupervised_clustering(self, X):
        """Phase 1: Discover learner clusters using K-Means."""
        
        print("\n" + "="*60)
        print("PHASE 1: UNSUPERVISED LEARNING (CLUSTERING)")
        print("="*60)
        
        # Step 1: Scale features
        print("\n📊 Step 1: Feature scaling...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 2: PCA for dimensionality reduction
        print(f"📊 Step 2: PCA dimensionality reduction...")
        X_pca = self.pca.fit_transform(X_scaled)
        n_components = X_pca.shape[1]
        variance_explained = self.pca.explained_variance_ratio_.sum()
        
        print(f"   Original dimensions: {X.shape[1]}")
        print(f"   Reduced dimensions: {n_components}")
        print(f"   Variance explained: {variance_explained:.1%}")
        
        # Step 3: K-Means clustering
        print(f"📊 Step 3: K-Means clustering (k={self.n_clusters})...")
        cluster_labels = self.kmeans.fit_predict(X_pca)
        
        print(f"   Inertia: {self.kmeans.inertia_:.2f}")
        
        # Cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print("\n   Cluster Distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"      Cluster {cluster_id}: {count:6,} learners ({count/len(X)*100:5.1f}%)")
        
        return cluster_labels, X_scaled
    
    def profile_clusters(self, X, cluster_labels, feature_names):
        """Analyze and name clusters based on behavioral patterns."""
        
        print("\n📊 Profiling clusters...")
        
        # Map cluster IDs to interpretable names
        cluster_names = {
            0: "Strugglers",
            1: "Procrastinators", 
            2: "Disengaged",
            3: "Overwhelmed",
            4: "Thrivers"
        }
        
        # Analyze each cluster
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_data = X[mask]
            
            profile = {
                'name': cluster_names.get(cluster_id, f"Cluster_{cluster_id}"),
                'size': mask.sum(),
                'characteristics': {}
            }
            
            # Key characteristics
            for feature in ['quiz_avg_score', 'video_completion_rate', 'login_frequency_per_week', 
                          'forum_post_count', 'engagement_score']:
                if feature in feature_names:
                    idx = feature_names.index(feature)
                    profile['characteristics'][feature] = cluster_data[:, idx].mean()
            
            self.cluster_profiles[cluster_id] = profile
            
            print(f"\n   {profile['name']} (n={profile['size']:,}):")
            print(f"      Quiz Score: {profile['characteristics'].get('quiz_avg_score', 0):.2f}")
            print(f"      Video Completion: {profile['characteristics'].get('video_completion_rate', 0):.2f}")
            print(f"      Login Freq: {profile['characteristics'].get('login_frequency_per_week', 0):.2f}")
    
    def phase2_supervised_classification(self, X_scaled, cluster_labels, y):
        """Phase 2: Train Random Forest per cluster for dropout prediction."""
        
        print("\n" + "="*60)
        print("PHASE 2: SUPERVISED LEARNING (PREDICTION)")
        print("="*60)
        
        # Train one model per cluster
        for cluster_id in range(self.n_clusters):
            cluster_name = self.cluster_profiles[cluster_id]['name']
            print(f"\n📊 Training model for {cluster_name}...")
            
            # Filter data for this cluster
            mask = cluster_labels == cluster_id
            X_cluster = X_scaled[mask]
            y_cluster = y[mask]
            
            if len(y_cluster) < 50:  # Skip if too few samples
                print(f"   ⚠️  Skipping (insufficient data: {len(y_cluster)} samples)")
                continue
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_cluster, y_cluster, test_size=0.2, random_state=self.random_state, stratify=y_cluster
            )
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',  # Handle imbalance
                random_state=self.random_state,
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            self.performance_metrics[cluster_id] = metrics
            self.rf_models[cluster_id] = rf
            
            print(f"   Accuracy:  {metrics['accuracy']:.3f}")
            print(f"   Precision: {metrics['precision']:.3f}")
            print(f"   Recall:    {metrics['recall']:.3f}")
            print(f"   F1-Score:  {metrics['f1']:.3f}")
            print(f"   AUC-ROC:   {metrics['auc_roc']:.3f}")
    
    def visualize_results(self, X_scaled, cluster_labels, output_dir='results'):
        """Create visualizations of clustering and model performance."""
        
        print("\n📊 Generating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Cluster Visualization (PCA 2D projection)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        X_pca_2d = PCA(n_components=2, random_state=self.random_state).fit_transform(X_scaled)
        
        scatter = ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                            c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        ax1.set_title('Learner Clusters (PCA Projection)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # 2. Cluster Size Distribution
        cluster_sizes = [self.cluster_profiles[i]['size'] for i in range(self.n_clusters)]
        cluster_names = [self.cluster_profiles[i]['name'] for i in range(self.n_clusters)]
        
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
        ax2.barh(cluster_names, cluster_sizes, color=colors)
        ax2.set_xlabel('Number of Learners')
        ax2.set_title('Cluster Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(cluster_sizes):
            ax2.text(v + 100, i, f'{v:,}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: cluster_analysis.png")
        
        # 3. Model Performance Comparison
        if self.performance_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            cluster_ids = list(self.performance_metrics.keys())
            cluster_labels_plot = [self.cluster_profiles[i]['name'] for i in cluster_ids]
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for idx, (metric_key, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
                ax = axes[idx // 2, idx % 2]
                values = [self.performance_metrics[i][metric_key] for i in cluster_ids]
                
                bars = ax.bar(cluster_labels_plot, values, color=colors[:len(cluster_ids)])
                ax.set_ylabel(metric_name)
                ax.set_ylim(0, 1.0)
                ax.set_title(f'{metric_name} by Cluster', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: model_performance.png")
        
        plt.close('all')
    
    def save_models(self, output_dir='models'):
        """Save all trained models and transformers."""
        
        print("\n💾 Saving models...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler, PCA, and K-Means
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        print("   ✅ Saved: scaler.pkl")
        
        joblib.dump(self.pca, os.path.join(output_dir, 'pca.pkl'))
        print("   ✅ Saved: pca.pkl")
        
        joblib.dump(self.kmeans, os.path.join(output_dir, 'kmeans.pkl'))
        print("   ✅ Saved: kmeans.pkl")
        
        # Save cluster profiles
        joblib.dump(self.cluster_profiles, os.path.join(output_dir, 'cluster_profiles.pkl'))
        print("   ✅ Saved: cluster_profiles.pkl")
        
        # Save Random Forest models
        for cluster_id, rf in self.rf_models.items():
            filename = f'rf_cluster_{cluster_id}.pkl'
            joblib.dump(rf, os.path.join(output_dir, filename))
            print(f"   ✅ Saved: {filename}")
    
    def print_summary(self):
        """Print training summary."""
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        print("\nCLUSTER PROFILES:")
        for cluster_id, profile in self.cluster_profiles.items():
            print(f"\n  {profile['name']} (Cluster {cluster_id}):")
            print(f"     Size: {profile['size']:,} learners")
        
        if self.performance_metrics:
            print("\nMODEL PERFORMANCE:")
            print(f"\n  {'Cluster':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
            print("  " + "-"*70)
            
            for cluster_id, metrics in self.performance_metrics.items():
                name = self.cluster_profiles[cluster_id]['name']
                print(f"  {name:<20} {metrics['accuracy']:>10.3f} {metrics['precision']:>10.3f} "
                      f"{metrics['recall']:>10.3f} {metrics['f1']:>10.3f} {metrics['auc_roc']:>10.3f}")
            
            # Overall average
            avg_metrics = {
                k: np.mean([m[k] for m in self.performance_metrics.values()])
                for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
            }
            print("  " + "-"*70)
            print(f"  {'AVERAGE':<20} {avg_metrics['accuracy']:>10.3f} {avg_metrics['precision']:>10.3f} "
                  f"{avg_metrics['recall']:>10.3f} {avg_metrics['f1']:>10.3f} {avg_metrics['auc_roc']:>10.3f}")


def main():
    """Main training pipeline."""
    
    print("\n" + "="*60)
    print("EDURETAIN - MODEL TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Load data
    print("📂 Loading dataset...")
    df = pd.read_csv('data/coursera_learner_data.csv')
    print(f"   Loaded: {len(df):,} learners\n")
    
    # Initialize trainer
    trainer = EduRetainTrainer(n_clusters=5, random_state=42)
    
    # Prepare features
    X, y, feature_names = trainer.prepare_features(df)
    
    # Phase 1: Clustering
    cluster_labels, X_scaled = trainer.phase1_unsupervised_clustering(X)
    trainer.profile_clusters(X.values, cluster_labels, list(X.columns))
    
    # Phase 2: Supervised learning
    trainer.phase2_supervised_classification(X_scaled, cluster_labels, y)
    
    # Visualize
    trainer.visualize_results(X_scaled, cluster_labels)
    
    # Save models
    trainer.save_models()
    
    # Summary
    trainer.print_summary()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\n✅ All models trained and saved successfully!")
    print("\nNext step: Run 'python src/predict.py' to make predictions\n")


if __name__ == "__main__":
    main()
