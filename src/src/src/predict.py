
"""
EduRetain Prediction System

Generates dropout risk predictions for learners using the hybrid model.
Provides personalized intervention recommendations.
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime


class EduRetainPredictor:
    """
    Dropout risk prediction system.
    
    Uses trained hybrid model to:
    1. Assign learners to behavioral clusters
    2. Predict 7-day dropout probability
    3. Recommend personalized interventions
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.load_models()
        
        # Intervention strategies per cluster
        self.interventions = {
            'Strugglers': {
                'primary': 'Provide prerequisite review materials and simplified explanations',
                'secondary': 'Recommend 1-on-1 tutoring sessions',
                'timing': 'Immediate'
            },
            'Procrastinators': {
                'primary': 'Send deadline reminders and create micro-goals',
                'secondary': 'Suggest study schedule builder tool',
                'timing': '3 days before next deadline'
            },
            'Disengaged': {
                'primary': 'Invite to peer study group and share success stories',
                'secondary': 'Personalized motivational message from instructor',
                'timing': 'Immediate'
            },
            'Overwhelmed': {
                'primary': 'Offer course pacing adjustment and break recommendations',
                'secondary': 'Connect with academic advisor',
                'timing': 'Within 24 hours'
            },
            'Thrivers': {
                'primary': 'Encourage peer mentoring and advanced materials',
                'secondary': 'Invite to beta test new features',
                'timing': 'Optional / Low priority'
            }
        }
    
    def load_models(self):
        """Load all trained models."""
        
        print("📂 Loading trained models...")
        
        try:
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.pca = joblib.load(os.path.join(self.model_dir, 'pca.pkl'))
            self.kmeans = joblib.load(os.path.join(self.model_dir, 'kmeans.pkl'))
            self.cluster_profiles = joblib.load(os.path.join(self.model_dir, 'cluster_profiles.pkl'))
            
            # Load Random Forest models
            self.rf_models = {}
            for cluster_id in range(len(self.cluster_profiles)):
                model_path = os.path.join(self.model_dir, f'rf_cluster_{cluster_id}.pkl')
                if os.path.exists(model_path):
                    self.rf_models[cluster_id] = joblib.load(model_path)
            
            print(f"   ✅ Loaded {len(self.rf_models)} Random Forest models")
            print(f"   ✅ Loaded clustering pipeline")
            
        except FileNotFoundError as e:
            print(f"   ❌ Error: Model files not found. Please run train_model.py first.")
            raise e
    
    def prepare_features(self, df):
        """Extract features for prediction."""
        
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
        X = X.fillna(X.median())
        
        return X
    
    def predict(self, df):
        """
        Generate predictions for all learners.
        
        Returns:
            DataFrame with cluster assignments, dropout probabilities, and interventions
        """
        
        print("\n🔮 Generating predictions...")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Step 1: Scale features
        X_scaled = self.scaler.transform(X)
        
        # Step 2: PCA transformation
        X_pca = self.pca.transform(X_scaled)
        
        # Step 3: Cluster assignment
        cluster_labels = self.kmeans.predict(X_pca)
        
        # Step 4: Dropout probability prediction
        predictions = []
        
        for idx, (cluster_id, x_scaled) in enumerate(zip(cluster_labels, X_scaled)):
            
            pred_data = {
                'learner_id': df.iloc[idx]['learner_id'],
                'cluster_id': cluster_id,
                'cluster_name': self.cluster_profiles[cluster_id]['name']
            }
            
            # Predict using cluster-specific model
            if cluster_id in self.rf_models:
                rf_model = self.rf_models[cluster_id]
                dropout_prob = rf_model.predict_proba([x_scaled])[0][1]
                pred_data['dropout_probability'] = dropout_prob
                
                # Risk level
                if dropout_prob < 0.30:
                    pred_data['risk_level'] = 'Low'
                elif dropout_prob < 0.60:
                    pred_data['risk_level'] = 'Medium'
                else:
                    pred_data['risk_level'] = 'High'
                
                # Feature importance (top 3 risk factors)
                feature_importance = rf_model.feature_importances_
                top_features_idx = np.argsort(feature_importance)[-3:][::-1]
                pred_data['top_risk_factors'] = [X.columns[i] for i in top_features_idx]
                
            else:
                pred_data['dropout_probability'] = np.nan
                pred_data['risk_level'] = 'Unknown'
                pred_data['top_risk_factors'] = []
            
            # Intervention recommendation
            cluster_name = pred_data['cluster_name']
            if cluster_name in self.interventions:
                intervention = self.interventions[cluster_name]
                pred_data['intervention_primary'] = intervention['primary']
                pred_data['intervention_secondary'] = intervention['secondary']
                pred_data['intervention_timing'] = intervention['timing']
            
            predictions.append(pred_data)
        
        predictions_df = pd.DataFrame(predictions)
        
        print(f"   ✅ Generated predictions for {len(predictions_df):,} learners")
        
        return predictions_df
    
    def identify_high_risk(self, predictions_df, threshold=0.60):
        """Identify high-risk learners requiring immediate intervention."""
        
        high_risk = predictions_df[predictions_df['dropout_probability'] >= threshold].copy()
        high_risk = high_risk.sort_values('dropout_probability', ascending=False)
        
        return high_risk
    
    def save_predictions(self, predictions_df, output_dir='results'):
        """Save predictions to CSV."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(filepath, index=False)
        
        print(f"\n💾 Predictions saved to: {filepath}")
        
        return filepath
    
    def generate_intervention_report(self, predictions_df, output_dir='results'):
        """Generate actionable intervention report for instructors."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, 'intervention_report.txt')
        
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EDURETAIN - INTERVENTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Learners Analyzed: {len(predictions_df):,}\n")
            
            risk_counts = predictions_df['risk_level'].value_counts()
            for risk_level in ['High', 'Medium', 'Low']:
                count = risk_counts.get(risk_level, 0)
                pct = count / len(predictions_df) * 100
                f.write(f"{risk_level} Risk: {count:,} learners ({pct:.1f}%)\n")
            
            f.write(f"\nAverage Dropout Probability: {predictions_df['dropout_probability'].mean():.1%}\n")
            
            # Cluster breakdown
            f.write("\n" + "=" * 70 + "\n")
            f.write("CLUSTER BREAKDOWN:\n")
            f.write("=" * 70 + "\n\n")
            
            cluster_summary = predictions_df.groupby('cluster_name').agg({
                'dropout_probability': ['count', 'mean', 'std']
            }).round(3)
            
            f.write(cluster_summary.to_string())
            
            # High-risk learners
            high_risk = self.identify_high_risk(predictions_df, threshold=0.60)
            
            f.write("\n\n" + "=" * 70 + "\n")
            f.write(f"HIGH-RISK LEARNERS (n={len(high_risk):,})\n")
            f.write("=" * 70 + "\n")
            f.write("\nREQUIRE IMMEDIATE INTERVENTION\n\n")
            
            if len(high_risk) > 0:
                for idx, row in high_risk.head(20).iterrows():
                    f.write(f"\nLearner ID: {row['learner_id']}\n")
                    f.write(f"  Cluster: {row['cluster_name']}\n")
                    f.write(f"  Dropout Risk: {row['dropout_probability']:.1%}\n")
                    f.write(f"  Primary Intervention: {row['intervention_primary']}\n")
                    f.write(f"  Timing: {row['intervention_timing']}\n")
                    f.write(f"  Top Risk Factors: {', '.join(row['top_risk_factors'][:3])}\n")
                
                if len(high_risk) > 20:
                    f.write(f"\n... and {len(high_risk) - 20:,} more high-risk learners\n")
            else:
                f.write("No high-risk learners identified.\n")
        
        print(f"📊 Intervention report saved to: {filepath}")
        
        return filepath


def main():
    """Main prediction pipeline."""
    
    print("\n" + "="*70)
    print("EDURETAIN - DROPOUT PREDICTION SYSTEM")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading learner data...")
    df = pd.read_csv('data/coursera_learner_data.csv')
    print(f"   Loaded: {len(df):,} learners\n")
    
    # Initialize predictor
    predictor = EduRetainPredictor(model_dir='models')
    
    # Generate predictions
    predictions_df = predictor.predict(df)
    
    # Save predictions
    predictor.save_predictions(predictions_df)
    
    # Generate intervention report
    predictor.generate_intervention_report(predictions_df)
    
    # Display high-risk summary
    print("\n" + "="*70)
    print("HIGH-RISK LEARNERS SUMMARY")
    print("="*70 + "\n")
    
    high_risk = predictor.identify_high_risk(predictions_df, threshold=0.60)
    
    print(f"Total High-Risk Learners: {len(high_risk):,}")
    print(f"Percentage of Total: {len(high_risk)/len(predictions_df)*100:.1f}%\n")
    
    if len(high_risk) > 0:
        print("Top 10 Highest Risk:")
        print("-" * 70)
        print(f"{'Learner ID':<12} {'Cluster':<18} {'Dropout Risk':<15} {'Intervention'}")
        print("-" * 70)
        
        for idx, row in high_risk.head(10).iterrows():
            intervention_short = row['intervention_primary'][:35] + "..." if len(row['intervention_primary']) > 35 else row['intervention_primary']
            print(f"{row['learner_id']:<12} {row['cluster_name']:<18} {row['dropout_probability']:>12.1%}   {intervention_short}")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print("\n✅ Predictions and intervention recommendations generated!")
    print("\nNext steps:")
    print("  1. Review: results/predictions.csv")
    print("  2. Review: results/intervention_report.txt")
    print("  3. Deploy interventions for high-risk learners\n")


if __name__ == "__main__":
    main()
