python
"""
Tests for interpretability and profiling modules
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.interpretability.profile_segments import SegmentProfiler

@pytest.fixture
def sample_clustered_data():
    """Create sample clustered data for testing"""
    np.random.seed(42)
    
    n_customers = 100
    
    data = pd.DataFrame({
        'CustomerID': range(1000, 1000 + n_customers),
        'Cluster': np.random.choice([0, 1, 2, 3], n_customers),
        'Recency': np.random.randint(1, 365, n_customers),
        'Frequency': np.random.randint(1, 50, n_customers),
        'Monetary': np.random.exponential(1000, n_customers),
        'AvgTransactionValue': np.random.exponential(100, n_customers),
        'CustomerLifetime': np.random.randint(30, 730, n_customers),
        'PurchaseFrequency': np.random.exponential(0.5, n_customers),
        'TotalTransactions': np.random.randint(1, 100, n_customers)
    })
    
    return data

@pytest.fixture
def segment_profiler(tmp_path):
    """Create SegmentProfiler with temporary directories"""
    # Create temporary directories
    data_dir = tmp_path / 'data' / 'features'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = tmp_path / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = tmp_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create profiler and override directories
    profiler = SegmentProfiler()
    profiler.data_dir = data_dir
    profiler.models_dir = models_dir
    profiler.results_dir = results_dir
    
    return profiler

class TestSegmentProfiler:
    """Test SegmentProfiler class"""
    
    def test_load_clustered_data(self, segment_profiler, sample_clustered_data, tmp_path):
        """Test loading clustered data"""
        # Save sample data
        data_path = segment_profiler.data_dir / 'customer_segments.parquet'
        sample_clustered_data.to_parquet(data_path)
        
        # Load the data
        loaded_data = segment_profiler.load_clustered_data()
        
        # Check data was loaded
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_clustered_data)
        assert 'Segment_Name' in loaded_data.columns
        
    def test_load_nonexistent_clustered_data(self, segment_profiler):
        """Test loading non-existent clustered data"""
        with pytest.raises(FileNotFoundError):
            segment_profiler.load_clustered_data()
            
    def test_analyze_segment_distribution(self, segment_profiler, sample_clustered_data):
        """Test segment distribution analysis"""
        # Add segment names
        sample_clustered_data['Segment_Name'] = sample_clustered_data['Cluster'].map(
            segment_profiler.segment_names
        )
        
        # Save to temporary location
        data_path = segment_profiler.data_dir / 'customer_segments.parquet'
        sample_clustered_data.to_parquet(data_path)
        
        # Load data
        loaded_data = segment_profiler.load_clustered_data()
        
        # Analyze distribution
        distribution = segment_profiler.analyze_segment_distribution(loaded_data)
        
        # Check structure
        assert 'total_customers' in distribution
        assert 'segments' in distribution
        
        # Check total customers
        assert distribution['total_customers'] == len(loaded_data)
        
        # Check each segment
        for segment_name, segment_info in distribution['segments'].items():
            assert 'cluster_id' in segment_info
            assert 'count' in segment_info
            assert 'percentage' in segment_info
            assert 'description' in segment_info
            
            # Check counts match
            segment_data = loaded_data[loaded_data['Segment_Name'] == segment_name]
            assert segment_info['count'] == len(segment_data)
            
            # Check percentage calculation
            expected_percentage = (segment_info['count'] / distribution['total_customers']) * 100
            assert pytest.approx(segment_info['percentage'], 0.01) == expected_percentage
            
    def test_analyze_segment_characteristics(self, segment_profiler, sample_clustered_data):
        """Test segment characteristics analysis"""
        # Add segment names
        sample_clustered_data['Segment_Name'] = sample_clustered_data['Cluster'].map(
            segment_profiler.segment_names
        )
        
        # Save to temporary location
        data_path = segment_profiler.data_dir / 'customer_segments.parquet'
        sample_clustered_data.to_parquet(data_path)
        
        # Load data
        loaded_data = segment_profiler.load_clustered_data()
        
        # Analyze characteristics
        characteristics = segment_profiler.analyze_segment_characteristics(loaded_data)
        
        # Check structure
        assert isinstance(characteristics, dict)
        
        # Check each segment
        for segment_name, segment_stats in characteristics.items():
            assert 'size' in segment_stats
            assert 'percentage' in segment_stats
            assert 'metrics' in segment_stats
            
            # Check metrics
            for metric, metric_stats in segment_stats['metrics'].items():
                assert 'mean' in metric_stats
                assert 'median' in metric_stats
                assert 'std' in metric_stats
                assert 'min' in metric_stats
                assert 'max' in metric_stats
                
                # Check statistical validity
                assert metric_stats['min'] <= metric_stats['mean'] <= metric_stats['max']
                assert metric_stats['min'] <= metric_stats['median'] <= metric_stats['max']
                
    def test_analyze_revenue_contribution(self, segment_profiler, sample_clustered_data):
        """Test revenue contribution analysis"""
        # Add segment names
        sample_clustered_data['Segment_Name'] = sample_clustered_data['Cluster'].map(
            segment_profiler.segment_names
        )
        
        # Save to temporary location
        data_path = segment_profiler.data_dir / 'customer_segments.parquet'
        sample_clustered_data.to_parquet(data_path)
        
        # Load data
        loaded_data = segment_profiler.load_clustered_data()
        
        # Analyze revenue
        revenue_analysis = segment_profiler.analyze_revenue_contribution(loaded_data)
        
        # Check structure
        assert isinstance(revenue_analysis, dict)
        
        # Check each segment
        for segment_name, revenue_info in revenue_analysis.items():
            assert 'Total_Revenue' in revenue_info
            assert 'Avg_Revenue_per_Customer' in revenue_info
            assert 'Customer_Count' in revenue_info
            assert 'Revenue_Percentage' in revenue_info
            assert 'Customer_Percentage' in revenue_info
            
            # Check calculations
            segment_data = loaded_data[loaded_data['Segment_Name'] == segment_name]
            
            # Total revenue should match sum of Monetary
            expected_revenue = segment_data['Monetary'].sum()
            assert pytest.approx(revenue_info['Total_Revenue'], 0.01) == expected_revenue
            
            # Customer count should match
            assert revenue_info['Customer_Count'] == len(segment_data)
            
            # Percentages should sum to ~100
            total_revenue = loaded_data['Monetary'].sum()
            expected_percentage = (expected_revenue / total_revenue) * 100
            assert pytest.approx(revenue_info['Revenue_Percentage'], 0.01) == expected_percentage
            
    def test_generate_business_recommendations(self, segment_profiler, sample_clustered_data):
        """Test business recommendations generation"""
        # Add segment names
        sample_clustered_data['Segment_Name'] = sample_clustered_data['Cluster'].map(
            segment_profiler.segment_names
        )
        
        # Save to temporary location
        data_path = segment_profiler.data_dir / 'customer_segments.parquet'
        sample_clustered_data.to_parquet(data_path)
        
        # Load data
        loaded_data = segment_profiler.load_clustered_data()
        
        # Generate recommendations
        recommendations = segment_profiler.generate_business_recommendations(loaded_data)
        
        # Check structure
        assert isinstance(recommendations, dict)
        
        # Check each segment
        for segment_name, segment_info in recommendations.items():
            assert 'description' in segment_info
            assert 'size' in segment_info
            assert 'key_characteristics' in segment_info
            assert 'business_actions' in segment_info
            assert 'success_metrics' in segment_info
            
            # Check business actions structure
            business_actions = segment_info['business_actions']
            assert 'goal' in business_actions
            assert 'actions' in business_actions
            assert 'kpis' in business_actions
            
            # Actions should be a list
            assert isinstance(business_actions['actions'], list)
            assert isinstance(business_actions['kpis'], list)
            
    def test_create_executive_summary(self, segment_profiler):
        """Test executive summary creation"""
        # Create mock analyses
        mock_analyses = {
            'distribution': {
                'total_customers': 1000,
                'segments': {
                    'VIP Customers': {'count': 200, 'percentage': 20.0},
                    'Regular Customers': {'count': 300, 'percentage': 30.0},
                    'At Risk Customers': {'count': 500, 'percentage': 50.0}
                }
            },
            'revenue': {
                'VIP Customers': {
                    'Total_Revenue': 500000,
                    'Revenue_Percentage': 50.0,
                    'Avg_Revenue_per_Customer': 2500
                },
                'Regular Customers': {
                    'Total_Revenue': 300000,
                    'Revenue_Percentage': 30.0,
                    'Avg_Revenue_per_Customer': 1000
                },
                'At Risk Customers': {
                    'Total_Revenue': 200000,
                    'Revenue_Percentage': 20.0,
                    'Avg_Revenue_per_Customer': 400
                }
            }
        }
        
        # Create summary
        summary = segment_profiler.create_executive_summary(mock_analyses)
        
        # Check structure
        assert 'project_overview' in summary
        assert 'key_findings' in summary
        assert 'business_impact' in summary
        assert 'recommendations' in summary
        assert 'next_steps' in summary
        
        # Check project overview
        overview = summary['project_overview']
        assert 'title' in overview
        assert 'date' in overview
        assert 'total_customers' in overview
        assert 'segments_identified' in overview
        
        # Check key findings
        assert isinstance(summary['key_findings'], list)
        assert len(summary['key_findings']) > 0
        
        # Check business impact
        assert isinstance(summary['business_impact'], list)
        
        # Check recommendations
        assert isinstance(summary['recommendations'], list)
        assert len(summary['recommendations']) > 0
        
        # Check next steps
        assert isinstance(summary['next_steps'], list)
        
    def test_generate_full_report(self, segment_profiler, sample_clustered_data, tmp_path):
        """Test full report generation"""
        # Add segment names
        sample_clustered_data['Segment_Name'] = sample_clustered_data['Cluster'].map(
            segment_profiler.segment_names
        )
        
        # Save to temporary location
        data_path = segment_profiler.data_dir / 'customer_segments.parquet'
        sample_clustered_data.to_parquet(data_path)
        
        # Generate full report
        report = segment_profiler.generate_full_report()
        
        # Check structure
        assert 'distribution' in report
        assert 'characteristics' in report
        assert 'revenue' in report
        assert 'recommendations' in report
        assert 'executive_summary' in report
        
        # Check report was saved
        report_path = segment_profiler.results_dir / 'segment_profiling_report.json'
        assert report_path.exists()
        
        # Load and verify saved report
        with open(report_path, 'r') as f:
            saved_report = json.load(f)
            
        assert isinstance(saved_report, dict)
        
    def test_segment_naming_mapping(self, segment_profiler):
        """Test segment name mapping"""
        # Test mapping from cluster ID to name
        assert segment_profiler.segment_names[0] == "VIP Customers"
        assert segment_profiler.segment_names[1] == "Occasional Shoppers"
        assert segment_profiler.segment_names[2] == "Regular Loyalists"
        assert segment_profiler.segment_names[3] == "At Risk Customers"
        
        # Test descriptions
        assert 0 in segment_profiler.segment_descriptions
        assert 1 in segment_profiler.segment_descriptions
        assert 2 in segment_profiler.segment_descriptions
        assert 3 in segment_profiler.segment_descriptions
        
        # Test business actions
        assert 0 in segment_profiler.segment_actions
        assert 1 in segment_profiler.segment_actions
        assert 2 in segment_profiler.segment_actions
        assert 3 in segment_profiler.segment_actions
        
    def test_get_cluster_id_from_name(self, segment_profiler):
        """Test getting cluster ID from segment name"""
        # Test valid names
        assert segment_profiler._get_cluster_id_from_name("VIP Customers") == 0
        assert segment_profiler._get_cluster_id_from_name("Occasional Shoppers") == 1
        assert segment_profiler._get_cluster_id_from_name("Regular Loyalists") == 2
        assert segment_profiler._get_cluster_id_from_name("At Risk Customers") == 3
        
        # Test invalid name
        assert segment_profiler._get_cluster_id_from_name("Non-existent Segment") is None
        
    def test_plot_generation(self, segment_profiler, sample_clustered_data, tmp_path):
        """Test plot generation (should not crash)"""
        # Add segment names
        sample_clustered_data['Segment_Name'] = sample_clustered_data['Cluster'].map(
            segment_profiler.segment_names
        )
        
        # Save to temporary location
        data_path = segment_profiler.data_dir / 'customer_segments.parquet'
        sample_clustered_data.to_parquet(data_path)
        
        # Load data
        loaded_data = segment_profiler.load_clustered_data()
        
        # These methods should generate plots without errors
        try:
            # Test distribution plot
            segment_profiler._plot_segment_distribution(
                loaded_data['Segment_Name'].value_counts(),
                len(loaded_data)
            )
            
            # Test comparison plot
            segment_profiler._plot_segment_comparison(
                loaded_data,
                ['Recency', 'Frequency', 'Monetary']
            )
            
            # Test revenue analysis plot (needs revenue dataframe)
            revenue_df = loaded_data.groupby('Segment_Name')['Monetary'].agg(['sum', 'mean', 'count'])
            revenue_df.columns = ['Total_Revenue', 'Avg_Revenue_per_Customer', 'Customer_Count']
            revenue_df['Revenue_Percentage'] = (revenue_df['Total_Revenue'] / revenue_df['Total_Revenue'].sum() * 100)
            
            segment_profiler._plot_revenue_analysis(revenue_df)
            
            # Check plots were saved
            plot_files = list(segment_profiler.results_dir.glob('*.png'))
            assert len(plot_files) > 0
            
        except Exception as e:
            pytest.fail(f"Plot generation failed: {str(e)}")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])