"""
Segment profiling module for customer segmentation system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentProfiler:
    """Profiles and interprets customer segments"""
    
    def __init__(self):
        """Initialize SegmentProfiler"""
        self.data_dir = Path('../data/features/')
        self.models_dir = Path('../models/')
        self.results_dir = Path('../results/')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Segment naming and descriptions
        self.segment_names = {
            0: "VIP Customers",
            1: "Occasional Shoppers",
            2: "Regular Loyalists",
            3: "At Risk Customers"
        }
        
        self.segment_descriptions = {
            0: "High-frequency, high-value customers who purchase regularly and recently",
            1: "Infrequent buyers with moderate recency and lower spend",
            2: "Customers with good purchase frequency and value but not recently active",
            3: "Customers who haven't purchased in a long time (dormant)"
        }
        
        # Business actions for each segment
        self.segment_actions = {
            0: {
                'goal': 'Retain and grow',
                'actions': [
                    'Exclusive loyalty program',
                    'Personalized recommendations',
                    'Early access to new products',
                    'Dedicated account manager'
                ],
                'kpis': [
                    'Increase purchase frequency by 20%',
                    'Increase average order value by 15%',
                    'Maintain churn rate below 5%'
                ]
            },
            1: {
                'goal': 'Increase frequency and loyalty',
                'actions': [
                    'Re-engagement campaigns',
                    'Cross-selling based on history',
                    'Loyalty program enrollment',
                    'Personalized win-back offers'
                ],
                'kpis': [
                    'Increase purchase frequency to 3+ per year',
                    'Convert 30% to Regular Loyalists',
                    'Increase retention rate by 25%'
                ]
            },
            2: {
                'goal': 'Re-activate and prevent churn',
                'actions': [
                    'Win-back campaigns',
                    'Survey to understand churn reasons',
                    'Product usage reminders',
                    'Re-activation bundles'
                ],
                'kpis': [
                    'Re-activate 40% of dormant customers',
                    'Reduce churn rate by 30%',
                    'Increase CLV by 20%'
                ]
            },
            3: {
                'goal': 'Win back or understand churn',
                'actions': [
                    'Aggressive win-back campaigns',
                    'Exit surveys',
                    'Special comeback offers',
                    'Review pain points'
                ],
                'kpis': [
                    'Win back 15% of churned customers',
                    'Reduce churn rate by 20%',
                    'Improve CSAT scores'
                ]
            }
        }
    
    def load_clustered_data(self):
        """
        Load clustered customer data
        
        Returns:
            DataFrame with cluster assignments
        """
        try:
            data_path = self.data_dir / 'customer_segments.parquet'
            clustered_data = pd.read_parquet(data_path)
            
            # Add segment names
            clustered_data['Segment_Name'] = clustered_data['Cluster'].map(self.segment_names)
            
            logger.info(f"Loaded {len(clustered_data)} clustered customers")
            return clustered_data
            
        except FileNotFoundError:
            logger.error("Clustered data not found. Run training first.")
            raise
    
    def analyze_segment_distribution(self, clustered_data):
        """
        Analyze segment size distribution
        
        Args:
            clustered_data: DataFrame with cluster assignments
            
        Returns:
            Dictionary with distribution analysis
        """
        logger.info("Analyzing segment distribution...")
        
        segment_counts = clustered_data['Segment_Name'].value_counts()
        total_customers = len(clustered_data)
        
        distribution = {
            'total_customers': total_customers,
            'segments': {}
        }
        
        for segment, count in segment_counts.items():
            percentage = (count / total_customers) * 100
            cluster_id = self._get_cluster_id_from_name(segment)
            
            distribution['segments'][segment] = {
                'cluster_id': cluster_id,
                'count': int(count),
                'percentage': float(percentage),
                'description': self.segment_descriptions.get(cluster_id, '')
            }
        
        # Create visualization
        self._plot_segment_distribution(segment_counts, total_customers)
        
        return distribution
    
    def _get_cluster_id_from_name(self, segment_name):
        """Get cluster ID from segment name"""
        for cluster_id, name in self.segment_names.items():
            if name == segment_name:
                return cluster_id
        return None
    
    def _plot_segment_distribution(self, segment_counts, total_customers):
        """Plot segment distribution"""
        plt.figure(figsize=(10, 6))
        bars = plt.bar(segment_counts.index, segment_counts.values, 
                      color=sns.color_palette("Set2", len(segment_counts)))
        
        plt.title('Customer Segment Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Customer Segment', fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for bar, segment in zip(bars, segment_counts.index):
            count = segment_counts[segment]
            percentage = (count / total_customers) * 100
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'segment_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved distribution plot to {plot_path}")
    
    def analyze_segment_characteristics(self, clustered_data):
        """
        Analyze characteristics of each segment
        
        Args:
            clustered_data: DataFrame with cluster assignments
            
        Returns:
            Dictionary with segment characteristics
        """
        logger.info("Analyzing segment characteristics...")
        
        # Key metrics to analyze
        metrics = ['Recency', 'Frequency', 'Monetary', 'AvgTransactionValue', 
                  'CustomerLifetime', 'PurchaseFrequency', 'TotalTransactions']
        
        characteristics = {}
        
        for cluster_id, segment_name in self.segment_names.items():
            segment_data = clustered_data[clustered_data['Cluster'] == cluster_id]
            
            segment_stats = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(clustered_data) * 100,
                'metrics': {}
            }
            
            # Calculate statistics for each metric
            for metric in metrics:
                if metric in segment_data.columns:
                    segment_stats['metrics'][metric] = {
                        'mean': float(segment_data[metric].mean()),
                        'median': float(segment_data[metric].median()),
                        'std': float(segment_data[metric].std()),
                        'min': float(segment_data[metric].min()),
                        'max': float(segment_data[metric].max()),
                        'q25': float(segment_data[metric].quantile(0.25)),
                        'q75': float(segment_data[metric].quantile(0.75))
                    }
            
            characteristics[segment_name] = segment_stats
        
        # Create comparative visualizations
        self._plot_segment_comparison(clustered_data, metrics)
        
        return characteristics
    
    def _plot_segment_comparison(self, clustered_data, metrics):
        """Create comparison plots for segments"""
        # Box plots for key metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        key_metrics = ['Recency', 'Frequency', 'Monetary', 
                      'AvgTransactionValue', 'CustomerLifetime', 'PurchaseFrequency']
        
        for idx, metric in enumerate(key_metrics[:6]):
            if metric in clustered_data.columns:
                data_to_plot = []
                segment_names = []
                
                for cluster_id in sorted(clustered_data['Cluster'].unique()):
                    segment_data = clustered_data[clustered_data['Cluster'] == cluster_id][metric]
                    if len(segment_data) > 0:
                        data_to_plot.append(segment_data)
                        segment_names.append(self.segment_names[cluster_id])
                
                axes[idx].boxplot(data_to_plot, labels=segment_names)
                axes[idx].set_title(f'{metric} by Segment')
                axes[idx].set_ylabel(metric)
                axes[idx].tick_params(axis='x', rotation=45)
                
                # Log scale for monetary metrics
                if metric in ['Monetary', 'AvgTransactionValue']:
                    axes[idx].set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'segment_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot to {plot_path}")
    
    def analyze_revenue_contribution(self, clustered_data):
        """
        Analyze revenue contribution by segment
        
        Args:
            clustered_data: DataFrame with cluster assignments
            
        Returns:
            Dictionary with revenue analysis
        """
        logger.info("Analyzing revenue contribution...")
        
        revenue_by_segment = clustered_data.groupby('Segment_Name')['Monetary'].agg(['sum', 'mean', 'median', 'count'])
        revenue_by_segment = revenue_by_segment.rename(columns={
            'sum': 'Total_Revenue',
            'mean': 'Avg_Revenue_per_Customer',
            'median': 'Median_Revenue_per_Customer',
            'count': 'Customer_Count'
        })
        
        # Calculate percentages
        total_revenue = revenue_by_segment['Total_Revenue'].sum()
        revenue_by_segment['Revenue_Percentage'] = (revenue_by_segment['Total_Revenue'] / total_revenue * 100).round(2)
        revenue_by_segment['Customer_Percentage'] = (revenue_by_segment['Customer_Count'] / len(clustered_data) * 100).round(2)
        
        # Calculate revenue per customer percentile
        revenue_by_segment['Revenue_per_Customer_Percentile'] = (
            revenue_by_segment['Avg_Revenue_per_Customer'] / revenue_by_segment['Avg_Revenue_per_Customer'].max() * 100
        ).round(2)
        
        # Create visualization
        self._plot_revenue_analysis(revenue_by_segment)
        
        return revenue_by_segment.to_dict('index')
    
    def _plot_revenue_analysis(self, revenue_df):
        """Plot revenue analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Revenue pie chart
        axes[0].pie(revenue_df['Revenue_Percentage'], labels=revenue_df.index, 
                   autopct='%1.1f%%', colors=sns.color_palette("Set2", len(revenue_df)))
        axes[0].set_title('Revenue Distribution by Segment', fontsize=14, fontweight='bold')
        
        # Customer count vs Revenue scatter
        scatter = axes[1].scatter(revenue_df['Customer_Count'], revenue_df['Total_Revenue'], 
                                 s=revenue_df['Avg_Revenue_per_Customer']/10, alpha=0.7)
        
        # Add labels
        for segment, row in revenue_df.iterrows():
            axes[1].annotate(segment, (row['Customer_Count'], row['Total_Revenue']),
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        axes[1].set_xlabel('Number of Customers')
        axes[1].set_ylabel('Total Revenue ($)')
        axes[1].set_title('Customer Count vs Revenue by Segment', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'revenue_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved revenue analysis plot to {plot_path}")
    
    def generate_business_recommendations(self, clustered_data):
        """
        Generate business recommendations for each segment
        
        Args:
            clustered_data: DataFrame with cluster assignments
            
        Returns:
            Dictionary with business recommendations
        """
        logger.info("Generating business recommendations...")
        
        recommendations = {}
        
        for cluster_id, segment_name in self.segment_names.items():
            segment_data = clustered_data[clustered_data['Cluster'] == cluster_id]
            
            segment_info = {
                'description': self.segment_descriptions.get(cluster_id, ''),
                'size': len(segment_data),
                'key_characteristics': {},
                'business_actions': self.segment_actions.get(cluster_id, {}),
                'success_metrics': [
                    f'Increase segment retention by 15%',
                    f'Improve segment revenue contribution by 10%',
                    f'Reduce segment churn rate by 20%'
                ]
            }
            
            # Add key characteristics
            if len(segment_data) > 0:
                segment_info['key_characteristics'] = {
                    'avg_recency': float(segment_data['Recency'].mean()),
                    'avg_frequency': float(segment_data['Frequency'].mean()),
                    'avg_monetary': float(segment_data['Monetary'].mean()),
                    'customer_lifetime': float(segment_data['CustomerLifetime'].mean()) if 'CustomerLifetime' in segment_data.columns else None
                }
            
            recommendations[segment_name] = segment_info
        
        return recommendations
    
    def create_executive_summary(self, analyses):
        """
        Create executive summary report
        
        Args:
            analyses: Dictionary containing all analyses
            
        Returns:
            Dictionary with executive summary
        """
        logger.info("Creating executive summary...")
        
        summary = {
            'project_overview': {
                'title': 'Customer Segmentation Analysis',
                'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'total_customers': analyses['distribution']['total_customers'],
                'segments_identified': len(analyses['distribution']['segments'])
            },
            'key_findings': [],
            'business_impact': [],
            'recommendations': [],
            'next_steps': []
        }
        
        # Extract key findings from distribution
        largest_segment = max(analyses['distribution']['segments'].items(), 
                             key=lambda x: x[1]['count'])
        smallest_segment = min(analyses['distribution']['segments'].items(), 
                              key=lambda x: x[1]['count'])
        
        summary['key_findings'].extend([
            f"Identified {summary['project_overview']['segments_identified']} distinct customer segments",
            f"Largest segment: {largest_segment[0]} ({largest_segment[1]['percentage']:.1f}% of customers)",
            f"Smallest segment: {smallest_segment[0]} ({smallest_segment[1]['percentage']:.1f}% of customers)"
        ])
        
        # Extract business impact from revenue analysis
        revenue_data = analyses['revenue']
        highest_revenue_segment = max(revenue_data.items(), 
                                     key=lambda x: x[1]['Total_Revenue'])
        
        summary['business_impact'].extend([
            f"Revenue concentration: {highest_revenue_segment[0]} contributes {highest_revenue_segment[1]['Revenue_Percentage']:.1f}% of total revenue",
            f"Customer value range: ${revenue_data[list(revenue_data.keys())[0]]['Avg_Revenue_per_Customer']:.2f} to ${revenue_data[list(revenue_data.keys())[-1]]['Avg_Revenue_per_Customer']:.2f} per customer"
        ])
        
        # Top recommendations
        summary['recommendations'].extend([
            "1. Focus retention efforts on highest-value segments",
            "2. Implement segment-specific marketing campaigns",
            "3. Develop targeted win-back strategies for at-risk segments",
            "4. Create personalized customer journeys for each segment"
        ])
        
        # Next steps
        summary['next_steps'].extend([
            "1. Implement recommendations in marketing automation",
            "2. Set up ongoing segment monitoring",
            "3. Quarterly review and model retraining",
            "4. A/B test segment-specific strategies"
        ])
        
        # Save summary
        summary_path = self.results_dir / 'executive_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved executive summary to {summary_path}")
        return summary
    
    def generate_full_report(self):
        """
        Generate full segment profiling report
        
        Returns:
            Dictionary with complete analysis
        """
        logger.info("Generating full segment profiling report...")
        
        # Load data
        clustered_data = self.load_clustered_data()
        
        # Run all analyses
        analyses = {
            'distribution': self.analyze_segment_distribution(clustered_data),
            'characteristics': self.analyze_segment_characteristics(clustered_data),
            'revenue': self.analyze_revenue_contribution(clustered_data),
            'recommendations': self.generate_business_recommendations(clustered_data)
        }
        
        # Create executive summary
        analyses['executive_summary'] = self.create_executive_summary(analyses)
        
        # Save complete report
        report_path = self.results_dir / 'segment_profiling_report.json'
        with open(report_path, 'w') as f:
            json.dump(analyses, f, indent=2)
        
        logger.info(f"Saved full report to {report_path}")
        
        # Print key insights
        self._print_key_insights(analyses)
        
        return analyses
    
    def _print_key_insights(self, analyses):
        """Print key insights to console"""
        print("\n" + "="*80)
        print("KEY INSIGHTS FROM SEGMENT PROFILING")
        print("="*80)
        
        # Distribution insights
        dist = analyses['distribution']
        print(f"\n📊 SEGMENT DISTRIBUTION:")
        for segment, info in dist['segments'].items():
            print(f"  • {segment}: {info['count']} customers ({info['percentage']:.1f}%)")
        
        # Revenue insights
        revenue = analyses['revenue']
        print(f"\n💰 REVENUE ANALYSIS:")
        for segment, info in revenue.items():
            print(f"  • {segment}: ${info['Total_Revenue']:,.0f} ({info['Revenue_Percentage']:.1f}% of total)")
        
        # Top recommendations
        print(f"\n🎯 TOP RECOMMENDATIONS:")
        for rec in analyses['executive_summary']['recommendations'][:3]:
            print(f"  {rec}")
        
        print("="*80)

# Example usage
if __name__ == "__main__":
    # Initialize profiler
    profiler = SegmentProfiler()
    
    # Generate full report
    report = profiler.generate_full_report()
    
    print(f"\nReport generated successfully!")
    print(f"Total customers analyzed: {report['distribution']['total_customers']}")
    print(f"Segments identified: {len(report['distribution']['segments'])}")
    
    # Example: Get specific segment info
    print(f"\nSegment details for 'VIP Customers':")
    if 'VIP Customers' in report['characteristics']:
        vip_info = report['characteristics']['VIP Customers']
        print(f"  Size: {vip_info['size']} customers")
        print(f"  Avg Monetary Value: ${vip_info['metrics']['Monetary']['mean']:.2f}")
        print(f"  Avg Frequency: {vip_info['metrics']['Frequency']['mean']:.1f} purchases")