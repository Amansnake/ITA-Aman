"""
Synthetic Dataset Generator for EduRetain Project

Generates realistic MOOC learner behavior data based on published research patterns.
Creates 50,000 learner records with 25+ features across 5 behavioral clusters.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

class CourseLearnerDataGenerator:
    """
    Generates synthetic learner data mimicking real Coursera patterns.
    
    Based on research from:
    - Kizilcec et al. (2013) on learner archetypes
    - Gardner & Brooks (2018) on dropout prediction
    """
    
    def __init__(self, n_learners=50000):
        self.n_learners = n_learners
        
        # Cluster distribution (from MOOC research)
        self.cluster_distribution = {
            'Strugglers': 0.22,
            'Procrastinators': 0.28,
            'Disengaged': 0.18,
            'Overwhelmed': 0.17,
            'Thrivers': 0.15
        }
        
        # Cluster-specific behavior patterns
        self.cluster_patterns = {
            'Strugglers': {
                'quiz_avg': (0.40, 0.50),
                'quiz_std': 0.12,
                'rewatch_rate': (0.55, 0.75),
                'video_completion': (0.50, 0.70),
                'forum_posts': (0, 2),
                'login_freq_per_week': (2.5, 4.5),
                'dropout_rate': 0.82
            },
            'Procrastinators': {
                'quiz_avg': (0.55, 0.70),
                'quiz_std': 0.15,
                'rewatch_rate': (0.25, 0.45),
                'video_completion': (0.60, 0.80),
                'forum_posts': (0, 3),
                'login_freq_per_week': (1.5, 3.0),
                'dropout_rate': 0.71
            },
            'Disengaged': {
                'quiz_avg': (0.45, 0.60),
                'quiz_std': 0.10,
                'rewatch_rate': (0.15, 0.35),
                'video_completion': (0.30, 0.50),
                'forum_posts': (0, 1),  # Changed from (0, 0) to (0, 1) for randint
                'login_freq_per_week': (0.5, 2.0),
                'dropout_rate': 0.87
            },
            'Overwhelmed': {
                'quiz_avg': (0.60, 0.75),
                'quiz_std': 0.18,
                'rewatch_rate': (0.30, 0.50),
                'video_completion': (0.65, 0.85),
                'forum_posts': (1, 5),
                'login_freq_per_week': (3.0, 5.5),
                'dropout_rate': 0.76
            },
            'Thrivers': {
                'quiz_avg': (0.75, 0.90),
                'quiz_std': 0.08,
                'rewatch_rate': (0.10, 0.25),
                'video_completion': (0.85, 0.98),
                'forum_posts': (3, 12),
                'login_freq_per_week': (4.5, 6.5),
                'dropout_rate': 0.12
            }
        }
    
    def generate_dataset(self):
        """Generate complete synthetic dataset."""
        
        print("🔧 Generating synthetic Coursera learner dataset...")
        print(f"📊 Total learners: {self.n_learners:,}")
        
        data = []
        learner_id = 1
        
        for cluster_name, proportion in self.cluster_distribution.items():
            n_cluster = int(self.n_learners * proportion)
            print(f"   Generating {cluster_name}: {n_cluster:,} learners...")
            
            for _ in range(n_cluster):
                learner_data = self._generate_learner(cluster_name, learner_id)
                data.append(learner_data)
                learner_id += 1
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle to mix clusters
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add learner_id as actual ID
        df['learner_id'] = range(1, len(df) + 1)
        
        print(f"✅ Dataset generated: {len(df):,} learners, {len(df.columns)} features")
        
        return df
    
    def _generate_learner(self, cluster, learner_id):
        """Generate a single learner's data based on cluster patterns."""
        
        pattern = self.cluster_patterns[cluster]
        
        # Basic info
        data = {
            'learner_id': learner_id,
            'cluster_true': cluster,  # Ground truth for validation
        }
        
        # Enrollment info
        data['course_id'] = np.random.choice(['ML101', 'DS201', 'PY301', 'AI401', 'WEB501'])
        data['enrollment_date'] = self._random_date()
        
        # Video engagement features
        data['video_completion_rate'] = np.clip(
            np.random.uniform(*pattern['video_completion']), 0, 1
        )
        data['rewatch_rate'] = np.clip(
            np.random.uniform(*pattern['rewatch_rate']), 0, 1
        )
        data['avg_playback_speed'] = np.random.choice(
            [1.0, 1.25, 1.5, 1.75, 2.0],
            p=[0.3, 0.25, 0.25, 0.15, 0.05]
        )
        data['video_pause_frequency'] = np.random.exponential(5.0) + 2
        data['skip_rate'] = np.clip(np.random.beta(2, 5), 0, 1)
        
        # Quiz/Assessment features
        quiz_avg = np.random.uniform(*pattern['quiz_avg'])
        data['quiz_avg_score'] = np.clip(quiz_avg, 0, 1)
        data['quiz_score_std'] = pattern['quiz_std']
        data['quiz_attempts_avg'] = np.clip(
            np.random.poisson(1.5) + 1, 1, 5
        )
        
        # Score improvement (Thrivers improve, Disengaged decline)
        if cluster == 'Thrivers':
            data['score_improvement'] = np.random.uniform(0.05, 0.20)
        elif cluster == 'Disengaged':
            data['score_improvement'] = np.random.uniform(-0.15, -0.05)
        else:
            data['score_improvement'] = np.random.uniform(-0.05, 0.10)
        
        data['assignment_submission_rate'] = np.clip(
            data['video_completion_rate'] + np.random.normal(0, 0.1), 0, 1
        )
        
        # Social engagement
        data['forum_post_count'] = np.random.randint(*pattern['forum_posts'])
        data['forum_reply_count'] = int(data['forum_post_count'] * np.random.uniform(0.5, 2.0))
        data['forum_upvotes_received'] = int(data['forum_post_count'] * np.random.uniform(0, 3))
        
        # Temporal/Engagement patterns
        data['login_frequency_per_week'] = np.clip(
            np.random.uniform(*pattern['login_freq_per_week']), 0.1, 10
        )
        data['avg_session_duration_min'] = np.clip(
            np.random.gamma(4, 15), 5, 180
        )
        data['days_since_last_login'] = self._generate_days_since_login(cluster)
        
        # Study pattern consistency
        if cluster == 'Procrastinators':
            data['study_time_variance'] = np.random.uniform(0.6, 0.9)  # High variance
        elif cluster == 'Thrivers':
            data['study_time_variance'] = np.random.uniform(0.1, 0.3)  # Low variance
        else:
            data['study_time_variance'] = np.random.uniform(0.3, 0.6)
        
        data['deadline_adherence_rate'] = np.clip(
            1.0 - data['study_time_variance'] + np.random.normal(0, 0.1), 0, 1
        )
        
        # Course progress
        data['percent_complete'] = self._generate_progress(cluster)
        data['weeks_enrolled'] = np.random.randint(1, 13)
        
        # Derived features
        data['engagement_score'] = self._calculate_engagement(data)
        data['performance_score'] = self._calculate_performance(data)
        
        # Target variable: Dropout (based on cluster dropout rate)
        data['dropped_out'] = int(np.random.random() < pattern['dropout_rate'])
        
        return data
    
    def _random_date(self):
        """Generate random enrollment date in past year."""
        start = datetime.now() - timedelta(days=365)
        random_days = np.random.randint(0, 365)
        return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')
    
    def _generate_days_since_login(self, cluster):
        """Generate days since last login based on cluster."""
        if cluster == 'Disengaged':
            return int(np.random.exponential(5) + 7)  # High value
        elif cluster == 'Thrivers':
            return int(np.random.exponential(1) + 1)  # Low value
        else:
            return int(np.random.exponential(3) + 2)
    
    def _generate_progress(self, cluster):
        """Generate course completion percentage based on cluster."""
        if cluster == 'Disengaged':
            return np.clip(np.random.beta(2, 5), 0, 0.4)
        elif cluster == 'Thrivers':
            return np.clip(np.random.beta(5, 2), 0.6, 1.0)
        elif cluster == 'Overwhelmed':
            # Start strong, then drop
            return np.clip(np.random.uniform(0.3, 0.6), 0, 1)
        else:
            return np.clip(np.random.beta(3, 3), 0, 1)
    
    def _calculate_engagement(self, data):
        """Calculate composite engagement score."""
        score = (
            data['video_completion_rate'] * 0.3 +
            data['login_frequency_per_week'] / 7.0 * 0.3 +
            data['forum_post_count'] / 15.0 * 0.2 +
            (1 - data['days_since_last_login'] / 30.0) * 0.2
        )
        return np.clip(score, 0, 1)
    
    def _calculate_performance(self, data):
        """Calculate composite performance score."""
        score = (
            data['quiz_avg_score'] * 0.5 +
            data['assignment_submission_rate'] * 0.3 +
            (data['score_improvement'] + 0.2) / 0.4 * 0.2  # Normalize to 0-1
        )
        return np.clip(score, 0, 1)
    
    def save_dataset(self, df, output_dir='data'):
        """Save dataset to CSV with summary statistics."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, 'coursera_learner_data.csv')
        df.to_csv(filepath, index=False)
        print(f"💾 Dataset saved to: {filepath}")
        
        # Generate summary
        summary_path = os.path.join(output_dir, 'data_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("COURSERA LEARNER DATASET - SUMMARY STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Learners: {len(df):,}\n")
            f.write(f"Features: {len(df.columns)}\n")
            f.write(f"Dropout Rate: {df['dropped_out'].mean():.1%}\n\n")
            
            f.write("CLUSTER DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            cluster_counts = df['cluster_true'].value_counts()
            for cluster, count in cluster_counts.items():
                dropout_rate = df[df['cluster_true'] == cluster]['dropped_out'].mean()
                f.write(f"{cluster:20s}: {count:6,} ({count/len(df)*100:5.1f}%) | Dropout: {dropout_rate:.1%}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("FEATURE STATISTICS:\n")
            f.write("=" * 60 + "\n\n")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            f.write(df[numeric_cols].describe().to_string())
            
        print(f"📊 Summary saved to: {summary_path}")
        
        return filepath


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print("EDURETAIN - SYNTHETIC DATASET GENERATOR")
    print("="*60 + "\n")
    
    # Generate dataset
    generator = CourseLearnerDataGenerator(n_learners=50000)
    df = generator.generate_dataset()
    
    # Save
    filepath = generator.save_dataset(df)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\n✅ Successfully generated {len(df):,} learner records")
    print(f"📁 Saved to: {filepath}")
    print("\nNext steps:")
    print("  1. Run: python src/train_model.py")
    print("  2. Run: python src/predict.py")
    print("\n")


if __name__ == "__main__":
    main()

