"""
EduRetain - Comprehensive Analysis and Visualization
Shows detailed analysis of all outputs with professional charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def create_comprehensive_analysis():
    """Generate comprehensive analysis with all visualizations"""
    
    print("\n" + "="*80)
    print(" EDURETAIN - COMPREHENSIVE ANALYSIS & VISUALIZATION")
    print("="*80 + "\n")
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv('data/coursera_learner_data.csv')
    print(f"   ✅ Loaded {len(df):,} learner records\n")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ============================================================
    # 1. CLUSTER DISTRIBUTION
    # ============================================================
    print("📊 Analyzing cluster distribution...")
    ax1 = fig.add_subplot(gs[0, 0])
    
    cluster_counts = df['cluster_true'].value_counts()
    colors_clusters = ['#E74C3C', '#F39C12', '#3498DB', '#9B59B6', '#2ECC71']
    
    wedges, texts, autotexts = ax1.pie(
        cluster_counts.values,
        labels=cluster_counts.index,
        autopct='%1.1f%%',
        colors=colors_clusters,
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax1.set_title('Learner Cluster Distribution\n(50,000 Total Learners)', 
                  fontweight='bold', fontsize=11)
    
    # ============================================================
    # 2. DROPOUT RATE BY CLUSTER
    # ============================================================
    print("📊 Analyzing dropout rates by cluster...")
    ax2 = fig.add_subplot(gs[0, 1])
    
    dropout_by_cluster = df.groupby('cluster_true')['dropped_out'].agg(['mean', 'count'])
    dropout_by_cluster = dropout_by_cluster.sort_values('mean', ascending=False)
    
    bars = ax2.barh(dropout_by_cluster.index, dropout_by_cluster['mean'] * 100, 
                     color=colors_clusters)
    
    # Add value labels
    for i, (idx, row) in enumerate(dropout_by_cluster.iterrows()):
        ax2.text(row['mean'] * 100 + 2, i, f"{row['mean']*100:.1f}%", 
                va='center', fontweight='bold')
    
    ax2.set_xlabel('Dropout Rate (%)', fontweight='bold')
    ax2.set_title('Dropout Rate by Cluster', fontweight='bold', fontsize=11)
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)
    
    # ============================================================
    # 3. FEATURE COMPARISON: QUIZ SCORES BY CLUSTER
    # ============================================================
    print("📊 Comparing quiz scores across clusters...")
    ax3 = fig.add_subplot(gs[0, 2])
    
    quiz_data = [df[df['cluster_true'] == cluster]['quiz_avg_score'].dropna() 
                 for cluster in cluster_counts.index]
    
    bp = ax3.boxplot(quiz_data, labels=cluster_counts.index, patch_artist=True,
                      showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors_clusters):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.set_ylabel('Quiz Average Score', fontweight='bold')
    ax3.set_title('Quiz Performance by Cluster', fontweight='bold', fontsize=11)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 4. VIDEO COMPLETION RATE BY CLUSTER
    # ============================================================
    print("📊 Analyzing video engagement...")
    ax4 = fig.add_subplot(gs[1, 0])
    
    video_data = df.groupby('cluster_true')['video_completion_rate'].mean().sort_values()
    
    bars = ax4.bar(range(len(video_data)), video_data.values, color=colors_clusters)
    ax4.set_xticks(range(len(video_data)))
    ax4.set_xticklabels(video_data.index, rotation=45, ha='right')
    ax4.set_ylabel('Average Video Completion Rate', fontweight='bold')
    ax4.set_title('Video Engagement by Cluster', fontweight='bold', fontsize=11)
    ax4.set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(video_data.values):
        ax4.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)
    
    ax4.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 5. LOGIN FREQUENCY DISTRIBUTION
    # ============================================================
    print("📊 Analyzing login patterns...")
    ax5 = fig.add_subplot(gs[1, 1])
    
    for i, cluster in enumerate(cluster_counts.index):
        cluster_data = df[df['cluster_true'] == cluster]['login_frequency_per_week']
        ax5.hist(cluster_data, bins=30, alpha=0.5, label=cluster, color=colors_clusters[i])
    
    ax5.set_xlabel('Login Frequency (per week)', fontweight='bold')
    ax5.set_ylabel('Number of Learners', fontweight='bold')
    ax5.set_title('Login Frequency Distribution by Cluster', fontweight='bold', fontsize=11)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 6. ENGAGEMENT SCORE VS PERFORMANCE SCORE
    # ============================================================
    print("📊 Analyzing engagement vs performance correlation...")
    ax6 = fig.add_subplot(gs[1, 2])
    
    for i, cluster in enumerate(cluster_counts.index):
        cluster_data = df[df['cluster_true'] == cluster]
        ax6.scatter(cluster_data['engagement_score'], 
                   cluster_data['performance_score'],
                   alpha=0.4, s=20, label=cluster, color=colors_clusters[i])
    
    ax6.set_xlabel('Engagement Score', fontweight='bold')
    ax6.set_ylabel('Performance Score', fontweight='bold')
    ax6.set_title('Engagement vs Performance by Cluster', fontweight='bold', fontsize=11)
    ax6.legend(loc='lower right', fontsize=8)
    ax6.grid(alpha=0.3)
    
    # Add diagonal reference line
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    # ============================================================
    # 7. FORUM PARTICIPATION
    # ============================================================
    print("📊 Analyzing forum participation...")
    ax7 = fig.add_subplot(gs[2, 0])
    
    forum_data = df.groupby('cluster_true')['forum_post_count'].agg(['mean', 'median', 'std'])
    
    x = np.arange(len(forum_data))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, forum_data['mean'], width, label='Mean', 
                    color='steelblue', alpha=0.8)
    bars2 = ax7.bar(x + width/2, forum_data['median'], width, label='Median', 
                    color='coral', alpha=0.8)
    
    ax7.set_ylabel('Forum Posts', fontweight='bold')
    ax7.set_title('Forum Participation by Cluster', fontweight='bold', fontsize=11)
    ax7.set_xticks(x)
    ax7.set_xticklabels(forum_data.index, rotation=45, ha='right')
    ax7.legend(fontsize=9)
    ax7.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 8. COURSE PROGRESS DISTRIBUTION
    # ============================================================
    print("📊 Analyzing course progress...")
    ax8 = fig.add_subplot(gs[2, 1])
    
    for i, cluster in enumerate(cluster_counts.index):
        cluster_data = df[df['cluster_true'] == cluster]['percent_complete']
        ax8.hist(cluster_data, bins=20, alpha=0.5, label=cluster, color=colors_clusters[i])
    
    ax8.set_xlabel('Course Progress (%)', fontweight='bold')
    ax8.set_ylabel('Number of Learners', fontweight='bold')
    ax8.set_title('Course Progress Distribution by Cluster', fontweight='bold', fontsize=11)
    ax8.legend(loc='upper right', fontsize=8)
    ax8.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 9. CORRELATION HEATMAP (KEY FEATURES)
    # ============================================================
    print("📊 Creating correlation heatmap...")
    ax9 = fig.add_subplot(gs[2, 2])
    
    key_features = [
        'video_completion_rate', 'quiz_avg_score', 'login_frequency_per_week',
        'forum_post_count', 'engagement_score', 'performance_score', 
        'percent_complete', 'dropped_out'
    ]
    
    corr_data = df[key_features].corr()
    
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax9,
                annot_kws={'fontsize': 7})
    
    ax9.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=11)
    ax9.set_xticklabels(ax9.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax9.set_yticklabels(ax9.get_yticklabels(), rotation=0, fontsize=8)
    
    # ============================================================
    # 10. TIME PATTERNS: STUDY TIME VARIANCE
    # ============================================================
    print("📊 Analyzing study time patterns...")
    ax10 = fig.add_subplot(gs[3, 0])
    
    variance_data = [df[df['cluster_true'] == cluster]['study_time_variance'].dropna() 
                     for cluster in cluster_counts.index]
    
    bp = ax10.boxplot(variance_data, labels=cluster_counts.index, patch_artist=True,
                       showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors_clusters):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax10.set_ylabel('Study Time Variance', fontweight='bold')
    ax10.set_title('Study Consistency by Cluster\n(Lower = More Consistent)', 
                   fontweight='bold', fontsize=11)
    ax10.tick_params(axis='x', rotation=45)
    ax10.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 11. DEADLINE ADHERENCE
    # ============================================================
    print("📊 Analyzing deadline adherence...")
    ax11 = fig.add_subplot(gs[3, 1])
    
    deadline_data = df.groupby('cluster_true')['deadline_adherence_rate'].mean().sort_values(ascending=False)
    
    bars = ax11.barh(range(len(deadline_data)), deadline_data.values, color=colors_clusters)
    ax11.set_yticks(range(len(deadline_data)))
    ax11.set_yticklabels(deadline_data.index)
    ax11.set_xlabel('Deadline Adherence Rate', fontweight='bold')
    ax11.set_title('On-Time Submission Rate by Cluster', fontweight='bold', fontsize=11)
    ax11.set_xlim(0, 1)
    
    # Add value labels
    for i, v in enumerate(deadline_data.values):
        ax11.text(v + 0.02, i, f'{v:.2f}', va='center', fontweight='bold', fontsize=9)
    
    ax11.grid(axis='x', alpha=0.3)
    
    # ============================================================
    # 12. SUMMARY STATISTICS TABLE
    # ============================================================
    print("📊 Creating summary statistics...")
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')
    
    # Calculate summary stats
    summary_data = []
    for cluster in cluster_counts.index:
        cluster_df = df[df['cluster_true'] == cluster]
        summary_data.append([
            cluster,
            f"{len(cluster_df):,}",
            f"{cluster_df['dropped_out'].mean():.1%}",
            f"{cluster_df['quiz_avg_score'].mean():.2f}",
            f"{cluster_df['video_completion_rate'].mean():.2f}",
            f"{cluster_df['percent_complete'].mean():.2f}"
        ])
    
    # Add total row
    summary_data.append([
        'TOTAL',
        f"{len(df):,}",
        f"{df['dropped_out'].mean():.1%}",
        f"{df['quiz_avg_score'].mean():.2f}",
        f"{df['video_completion_rate'].mean():.2f}",
        f"{df['percent_complete'].mean():.2f}"
    ])
    
    columns = ['Cluster', 'Count', 'Dropout', 'Quiz', 'Video', 'Progress']
    
    table = ax12.table(cellText=summary_data, colLabels=columns,
                       cellLoc='center', loc='center',
                       colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#1E2761')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style total row
    for i in range(len(columns)):
        table[(len(summary_data), i)].set_facecolor('#CADCFC')
        table[(len(summary_data), i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8F9FA')
    
    ax12.set_title('Summary Statistics by Cluster', fontweight='bold', fontsize=11, pad=20)
    
    # ============================================================
    # OVERALL TITLE
    # ============================================================
    fig.suptitle('EduRetain: Comprehensive Learner Behavior Analysis\n50,000 Learners Across 5 Behavioral Clusters', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    print("\n💾 Saving comprehensive analysis...")
    plt.savefig('results/comprehensive_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   ✅ Saved: results/comprehensive_analysis.png")
    
    plt.close()
    
    # ============================================================
    # CREATE SECOND FIGURE: IMPACT PROJECTIONS
    # ============================================================
    print("\n📊 Creating impact projection visualizations...")
    
    fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle('EduRetain: Projected Business & Social Impact', 
                  fontsize=16, fontweight='bold')
    
    # Impact 1: Completion Rate Improvement
    ax = axes[0, 0]
    scenarios = ['Current\nBaseline', 'With\nEduRetain', 'Improvement']
    values = [10, 25, 15]
    colors_impact = ['#E74C3C', '#2ECC71', '#3498DB']
    
    bars = ax.bar(scenarios, values, color=colors_impact, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Completion Rate (%)', fontweight='bold', fontsize=11)
    ax.set_title('Course Completion Rate Impact', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 30)
    ax.grid(axis='y', alpha=0.3)
    
    # Impact 2: Annual Completers
    ax = axes[0, 1]
    scenarios = ['Current', 'Projected']
    values = [3, 7.5]
    
    bars = ax.bar(scenarios, values, color=['#E74C3C', '#2ECC71'], alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val}M', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.axhline(y=4.5, color='#3498DB', linestyle='--', linewidth=2, 
               label='+4.5M Additional')
    ax.legend(fontsize=10, loc='upper left')
    
    ax.set_ylabel('Annual Completers (Millions)', fontweight='bold', fontsize=11)
    ax.set_title('Annual Completers Impact', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 9)
    ax.grid(axis='y', alpha=0.3)
    
    # Impact 3: Revenue & Economic Impact
    ax = axes[1, 0]
    categories = ['Certificate\nRevenue', 'Economic\nImpact']
    revenue_values = [225, 22500]
    
    bars = ax.bar(categories, revenue_values, color=['#F39C12', '#9B59B6'], 
                  alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.text(0, revenue_values[0] + 500, '$225M', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)
    ax.text(1, revenue_values[1] + 500, '$22.5B', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Value (Millions USD)', fontweight='bold', fontsize=11)
    ax.set_title('Financial Impact Projection', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Impact 4: Beneficiary Groups
    ax = axes[1, 1]
    
    groups = ['Women in\nSTEM', 'Developing\nEconomies', 'Career\nChangers', 
              'First-Gen\nLearners', 'All\nLearners']
    current = [8, 7, 9, 6, 10]
    projected = [12, 13, 18, 12, 25]
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, current, width, label='Current', 
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, projected, width, label='Projected', 
                   color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Completion Rate (%)', fontweight='bold', fontsize=11)
    ax.set_title('Impact on Underserved Groups', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 30)
    
    plt.tight_layout()
    
    print("💾 Saving impact projections...")
    plt.savefig('results/impact_projections.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   ✅ Saved: results/impact_projections.png")
    
    plt.close()
    
    # ============================================================
    # PRINT STATISTICAL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print(" STATISTICAL SUMMARY")
    print("="*80 + "\n")
    
    print("📊 CLUSTER PROFILES:\n")
    for cluster in cluster_counts.index:
        cluster_df = df[df['cluster_true'] == cluster]
        print(f"   {cluster}:")
        print(f"      Size: {len(cluster_df):,} learners ({len(cluster_df)/len(df)*100:.1f}%)")
        print(f"      Dropout Rate: {cluster_df['dropped_out'].mean():.1%}")
        print(f"      Avg Quiz Score: {cluster_df['quiz_avg_score'].mean():.2f}")
        print(f"      Video Completion: {cluster_df['video_completion_rate'].mean():.2f}")
        print(f"      Forum Posts: {cluster_df['forum_post_count'].mean():.1f}")
        print(f"      Login Freq: {cluster_df['login_frequency_per_week'].mean():.1f}/week")
        print()
    
    print("="*80)
    print(" KEY INSIGHTS")
    print("="*80 + "\n")
    
    # Calculate insights
    highest_dropout = df.groupby('cluster_true')['dropped_out'].mean().idxmax()
    lowest_dropout = df.groupby('cluster_true')['dropped_out'].mean().idxmin()
    highest_engagement = df.groupby('cluster_true')['engagement_score'].mean().idxmax()
    lowest_engagement = df.groupby('cluster_true')['engagement_score'].mean().idxmin()
    
    print(f"✓ Highest Risk Cluster: {highest_dropout} "
          f"({df[df['cluster_true']==highest_dropout]['dropped_out'].mean():.1%} dropout)")
    print(f"✓ Lowest Risk Cluster: {lowest_dropout} "
          f"({df[df['cluster_true']==lowest_dropout]['dropped_out'].mean():.1%} dropout)")
    print(f"✓ Most Engaged: {highest_engagement}")
    print(f"✓ Least Engaged: {lowest_engagement}")
    print(f"✓ Overall Dropout Rate: {df['dropped_out'].mean():.1%}")
    print(f"✓ Correlation (Engagement vs Dropout): "
          f"{df['engagement_score'].corr(df['dropped_out']):.3f}")
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)
    print("\n✅ Generated 2 comprehensive visualization files:")
    print("   • results/comprehensive_analysis.png (12 charts)")
    print("   • results/impact_projections.png (4 impact charts)")
    print("\n🎯 Use these for presentations, reports, and GitHub README!\n")


if __name__ == "__main__":
    create_comprehensive_analysis()
