"""
EduRetain - Main Pipeline Orchestrator

Runs the complete workflow:
1. Generate synthetic dataset
2. Train hybrid ML models
3. Generate predictions
4. Create evaluation reports
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(__file__))

from generate_data import CourseLearnerDataGenerator
from train_model import EduRetainTrainer
from predict import EduRetainPredictor
import pandas as pd


def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def run_full_pipeline():
    """Execute complete EduRetain pipeline."""
    
    print_header("EDURETAIN - COMPLETE PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ===== STEP 1: DATA GENERATION =====
    print_header("STEP 1: SYNTHETIC DATA GENERATION")
    
    print("Generating realistic learner behavior data...")
    generator = CourseLearnerDataGenerator(n_learners=50000)
    df = generator.generate_dataset()
    dataset_path = generator.save_dataset(df)
    
    print(f"\n✅ Dataset created: {len(df):,} learners")
    
    # ===== STEP 2: MODEL TRAINING =====
    print_header("STEP 2: HYBRID MODEL TRAINING")
    
    print("Training unsupervised clustering + supervised classification...")
    
    trainer = EduRetainTrainer(n_clusters=5, random_state=42)
    
    # Prepare features
    X, y, feature_names = trainer.prepare_features(df)
    
    # Phase 1: Clustering
    cluster_labels, X_scaled = trainer.phase1_unsupervised_clustering(X)
    trainer.profile_clusters(X.values, cluster_labels, list(X.columns))
    
    # Phase 2: Supervised learning
    trainer.phase2_supervised_classification(X_scaled, cluster_labels, y)
    
    # Visualize and save
    trainer.visualize_results(X_scaled, cluster_labels)
    trainer.save_models()
    trainer.print_summary()
    
    print("\n✅ Models trained and saved")
    
    # ===== STEP 3: PREDICTION =====
    print_header("STEP 3: DROPOUT RISK PREDICTION")
    
    print("Generating predictions for all learners...")
    
    predictor = EduRetainPredictor(model_dir='models')
    predictions_df = predictor.predict(df)
    predictor.save_predictions(predictions_df)
    predictor.generate_intervention_report(predictions_df)
    
    print("\n✅ Predictions generated")
    
    # ===== STEP 4: SUMMARY =====
    print_header("PIPELINE COMPLETE - SUMMARY")
    
    # Risk distribution
    risk_counts = predictions_df['risk_level'].value_counts()
    
    print("RISK DISTRIBUTION:")
    print("-" * 50)
    for risk_level in ['High', 'Medium', 'Low']:
        count = risk_counts.get(risk_level, 0)
        pct = count / len(predictions_df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {risk_level:<8}: {count:>7,} ({pct:>5.1f}%)  {bar}")
    
    # Cluster distribution
    print("\nCLUSTER DISTRIBUTION:")
    print("-" * 50)
    cluster_counts = predictions_df['cluster_name'].value_counts()
    for cluster_name, count in cluster_counts.items():
        pct = count / len(predictions_df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cluster_name:<16}: {count:>7,} ({pct:>5.1f}%)  {bar}")
    
    # Performance metrics
    print("\nMODEL PERFORMANCE:")
    print("-" * 50)
    
    if trainer.performance_metrics:
        avg_metrics = {
            k: sum(m[k] for m in trainer.performance_metrics.values()) / len(trainer.performance_metrics)
            for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        }
        
        print(f"  Average Accuracy:  {avg_metrics['accuracy']:.1%}")
        print(f"  Average Precision: {avg_metrics['precision']:.1%}")
        print(f"  Average Recall:    {avg_metrics['recall']:.1%}")
        print(f"  Average F1-Score:  {avg_metrics['f1']:.1%}")
        print(f"  Average AUC-ROC:   {avg_metrics['auc_roc']:.1%}")
    
    # Impact projections
    print("\nPROJECTED IMPACT:")
    print("-" * 50)
    
    baseline_completion = 0.10
    projected_completion = 0.25  # 15% improvement
    annual_enrollments = 30_000_000
    
    baseline_completers = int(annual_enrollments * baseline_completion)
    projected_completers = int(annual_enrollments * projected_completion)
    additional_completers = projected_completers - baseline_completers
    
    revenue_per_cert = 50
    additional_revenue = additional_completers * revenue_per_cert / 1_000_000
    
    print(f"  Baseline Completers:    {baseline_completers:>12,} (10%)")
    print(f"  Projected Completers:   {projected_completers:>12,} (25%)")
    print(f"  Additional Completers:  {additional_completers:>12,} (+15%)")
    print(f"  Additional Revenue:     ${additional_revenue:>11.0f}M")
    
    # Output files
    print("\nOUTPUT FILES:")
    print("-" * 50)
    print("  📄 data/coursera_learner_data.csv")
    print("  📊 results/cluster_analysis.png")
    print("  📊 results/model_performance.png")
    print("  📋 results/predictions.csv")
    print("  📋 results/intervention_report.txt")
    print("  🤖 models/*.pkl (8 model files)")
    
    print("\n" + "="*80)
    print("SUCCESS! EduRetain pipeline completed successfully.".center(80))
    print("="*80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("  • Review intervention_report.txt for action items")
    print("  • Deploy interventions for high-risk learners")
    print("  • Monitor impact on completion rates\n")


if __name__ == "__main__":
    try:
        run_full_pipeline()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

