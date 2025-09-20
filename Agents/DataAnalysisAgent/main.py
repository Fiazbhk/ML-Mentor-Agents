"""
DataAnalysisAgent - Analyzes datasets and provides insights for ML model selection
Part of the ML Mentor system for making machine learning accessible to beginners
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalysisAgent:
    """
    Intelligent agent that analyzes datasets and provides comprehensive insights
    for machine learning model selection and preprocessing recommendations.
    """
    
    def __init__(self):
        self.name = "DataAnalysisAgent"
        self.version = "1.0.0"
        self.description = "Analyzes datasets and provides ML-focused insights"
        self.supported_formats = ['.csv', '.json', '.xlsx', '.xls']
        
    async def analyze_dataset(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method that orchestrates all analysis tasks
        
        Args:
            input_data: Dictionary containing dataset info and parameters
            
        Returns:
            Comprehensive analysis results
        """
        try:
            logger.info(f"Starting dataset analysis at {datetime.now()}")
            
            # Load the dataset
            df = await self._load_dataset(input_data)
            
            if df is None or df.empty:
                return {"error": "Failed to load dataset or dataset is empty"}
            
            # Perform comprehensive analysis
            analysis_results = {
                "metadata": self._get_dataset_metadata(df),
                "shape_analysis": self._analyze_shape(df),
                "feature_analysis": self._analyze_features(df),
                "data_quality": self._analyze_data_quality(df),
                "statistical_summary": self._generate_statistical_summary(df),
                "missing_values": self._analyze_missing_values(df),
                "preprocessing_recommendations": self._suggest_preprocessing(df),
                "ml_insights": self._generate_ml_insights(df),
                "visualization_suggestions": self._suggest_visualizations(df)
            }
            
            # Calculate overall data quality score
            analysis_results["overall_quality_score"] = self._calculate_quality_score(analysis_results)
            
            # Generate human-readable summary
            analysis_results["summary"] = self._generate_summary(analysis_results)
            
            logger.info("Dataset analysis completed successfully")
            return {"success": True, "analysis": analysis_results}
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _load_dataset(self, input_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load dataset from various sources and formats"""
        try:
            if "file_content" in input_data:
                # Handle base64 encoded file content
                content = base64.b64decode(input_data["file_content"])
                file_type = input_data.get("file_type", "csv").lower()
                
                if file_type == "csv":
                    return pd.read_csv(io.BytesIO(content))
                elif file_type in ["xlsx", "xls"]:
                    return pd.read_excel(io.BytesIO(content))
                elif file_type == "json":
                    return pd.read_json(io.BytesIO(content))
                    
            elif "file_path" in input_data:
                # Handle file path
                file_path = Path(input_data["file_path"])
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                if file_path.suffix == ".csv":
                    return pd.read_csv(file_path)
                elif file_path.suffix in [".xlsx", ".xls"]:
                    return pd.read_excel(file_path)
                elif file_path.suffix == ".json":
                    return pd.read_json(file_path)
                    
            elif "dataframe" in input_data:
                # Handle direct dataframe input
                return input_data["dataframe"]
                
            else:
                raise ValueError("No valid dataset source provided")
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return None
    
    def _get_dataset_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic metadata about the dataset"""
        return {
            "name": "Unknown Dataset",
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_shape(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset dimensions and basic structure"""
        return {
            "dimensions": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "size_category": self._categorize_dataset_size(df.shape[0]),
            "aspect_ratio": round(df.shape[1] / df.shape[0], 4) if df.shape[0] > 0 else 0,
            "column_names": df.columns.tolist()
        }
    
    def _categorize_dataset_size(self, num_rows: int) -> str:
        """Categorize dataset size for ML context"""
        if num_rows < 100:
            return "very_small"
        elif num_rows < 1000:
            return "small"
        elif num_rows < 10000:
            return "medium"
        elif num_rows < 100000:
            return "large"
        else:
            return "very_large"
    
    def _analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive feature analysis"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()
        
        feature_analysis = {
            "types": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "datetime": datetime_columns,
                "boolean": boolean_columns
            },
            "counts": {
                "numeric": len(numeric_columns),
                "categorical": len(categorical_columns),
                "datetime": len(datetime_columns),
                "boolean": len(boolean_columns)
            }
        }
        
        # Analyze each feature type in detail
        if numeric_columns:
            feature_analysis["numeric_details"] = self._analyze_numeric_features(df[numeric_columns])
        
        if categorical_columns:
            feature_analysis["categorical_details"] = self._analyze_categorical_features(df[categorical_columns])
        
        return feature_analysis
    
    def _analyze_numeric_features(self, numeric_df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed analysis of numeric features"""
        analysis = {}
        
        for column in numeric_df.columns:
            col_data = numeric_df[column]
            analysis[column] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "outliers_count": self._count_outliers(col_data),
                "distribution_type": self._classify_distribution(col_data)
            }
        
        return analysis
    
    def _analyze_categorical_features(self, categorical_df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed analysis of categorical features"""
        analysis = {}
        
        for column in categorical_df.columns:
            col_data = categorical_df[column]
            value_counts = col_data.value_counts()
            
            analysis[column] = {
                "unique_values": int(col_data.nunique()),
                "most_frequent": value_counts.index[0] if not value_counts.empty else None,
                "most_frequent_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "cardinality": "high" if col_data.nunique() > 20 else "medium" if col_data.nunique() > 5 else "low",
                "top_5_values": value_counts.head(5).to_dict()
            }
        
        return analysis
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(((series < lower_bound) | (series > upper_bound)).sum())
    
    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify the distribution type of a numeric series"""
        skewness = series.skew()
        
        if abs(skewness) < 0.5:
            return "normal"
        elif skewness > 0.5:
            return "right_skewed"
        else:
            return "left_skewed"
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        quality_analysis = {
            "completeness": {
                "missing_cells": int(missing_cells),
                "total_cells": int(total_cells),
                "completeness_ratio": round((total_cells - missing_cells) / total_cells, 4),
                "columns_with_missing": df.columns[df.isnull().any()].tolist()
            },
            "consistency": {
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_ratio": round(df.duplicated().sum() / len(df), 4)
            },
            "validity": self._check_data_validity(df)
        }
        
        return quality_analysis
    
    def _check_data_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity issues"""
        validity_issues = {}
        
        for column in df.columns:
            col_issues = []
            
            # Check for mixed types (only for object columns)
            if df[column].dtype == 'object':
                sample_types = df[column].dropna().apply(type).unique()
                if len(sample_types) > 1:
                    col_issues.append("mixed_types")
            
            # Check for negative values in potentially positive-only columns
            if df[column].dtype in ['int64', 'float64']:
                if any(word in column.lower() for word in ['age', 'price', 'count', 'amount']):
                    if (df[column] < 0).any():
                        col_issues.append("negative_values_in_positive_field")
            
            if col_issues:
                validity_issues[column] = col_issues
        
        return validity_issues
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"message": "No numeric columns for statistical summary"}
        
        return {
            "describe": numeric_df.describe().to_dict(),
            "correlation_matrix": numeric_df.corr().to_dict() if len(numeric_df.columns) > 1 else {}
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed missing value analysis"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            "columns_with_missing": {
                col: {
                    "count": int(missing_counts[col]),
                    "percentage": round(missing_percentages[col], 2)
                }
                for col in missing_counts[missing_counts > 0].index
            },
            "patterns": self._identify_missing_patterns(df)
        }
    
    def _identify_missing_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify patterns in missing data"""
        patterns = []
        
        # Check for completely missing columns
        completely_missing = df.columns[df.isnull().all()].tolist()
        if completely_missing:
            patterns.append(f"Completely missing columns: {completely_missing}")
        
        # Check for rows with excessive missing values
        missing_per_row = df.isnull().sum(axis=1)
        high_missing_rows = (missing_per_row > len(df.columns) * 0.5).sum()
        if high_missing_rows > 0:
            patterns.append(f"{high_missing_rows} rows missing >50% of values")
        
        return patterns
    
    def _suggest_preprocessing(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate preprocessing recommendations"""
        recommendations = []
        
        # Missing value handling
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            recommendations.append({
                "type": "missing_values",
                "priority": "high",
                "description": "Handle missing values before model training",
                "suggestions": [
                    "Numeric columns: Consider median/mean imputation",
                    "Categorical columns: Consider mode imputation or 'unknown' category",
                    "Consider dropping columns with >80% missing values"
                ],
                "affected_columns": missing_cols
            })
        
        # Duplicate handling
        if df.duplicated().sum() > 0:
            recommendations.append({
                "type": "duplicates",
                "priority": "medium",
                "description": f"Remove {df.duplicated().sum()} duplicate rows",
                "suggestions": ["Use df.drop_duplicates() to remove duplicate rows"]
            })
        
        # Feature scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols and len(numeric_cols) > 1:
            # Check if scales are very different
            scales_differ = False
            for col in numeric_cols:
                if df[col].std() > 1000 or df[col].max() - df[col].min() > 1000:
                    scales_differ = True
                    break
            
            if scales_differ:
                recommendations.append({
                    "type": "scaling",
                    "priority": "medium",
                    "description": "Features have different scales",
                    "suggestions": [
                        "Consider StandardScaler for normal distributions",
                        "Consider MinMaxScaler for bounded features",
                        "Consider RobustScaler if outliers are present"
                    ]
                })
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            recommendations.append({
                "type": "encoding",
                "priority": "high",
                "description": "Encode categorical variables for ML algorithms",
                "suggestions": [
                    "Low cardinality (<10 categories): One-hot encoding",
                    "High cardinality (>10 categories): Label encoding or target encoding",
                    "Ordinal categories: Use ordinal encoding"
                ],
                "affected_columns": categorical_cols
            })
        
        return recommendations
    
    def _generate_ml_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights specific to machine learning"""
        insights = {
            "problem_type": self._suggest_problem_type(df),
            "dataset_suitability": self._assess_ml_suitability(df),
            "feature_importance_hints": self._analyze_feature_relationships(df)
        }
        
        return insights
    
    def _suggest_problem_type(self, df: pd.DataFrame) -> List[str]:
        """Suggest possible ML problem types based on data characteristics"""
        suggestions = []
        
        # Check for potential target variables
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() == 2:
                    suggestions.append(f"Binary classification (target: {col})")
                elif df[col].nunique() < 10 and df[col].nunique() > 2:
                    suggestions.append(f"Multi-class classification (target: {col})")
                elif df[col].nunique() > 10:
                    suggestions.append(f"Regression (target: {col})")
            
            elif df[col].dtype in ['object', 'category']:
                if df[col].nunique() < 10:
                    suggestions.append(f"Classification (target: {col})")
        
        if not suggestions:
            suggestions.append("Unsupervised learning (clustering, dimensionality reduction)")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _assess_ml_suitability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess how suitable the dataset is for ML"""
        suitability = {
            "sample_size": "good" if len(df) >= 1000 else "limited" if len(df) >= 100 else "very_limited",
            "feature_count": "good" if len(df.columns) >= 5 else "limited",
            "data_quality": "good" if df.isnull().sum().sum() / df.size < 0.1 else "needs_cleaning"
        }
        
        # Overall assessment
        scores = {"good": 1, "limited": 0.5, "needs_cleaning": 0.3, "very_limited": 0.1}
        avg_score = sum(scores.get(v, 0.5) for v in suitability.values()) / len(suitability)
        
        if avg_score >= 0.8:
            suitability["overall"] = "excellent"
        elif avg_score >= 0.6:
            suitability["overall"] = "good"
        elif avg_score >= 0.4:
            suitability["overall"] = "fair"
        else:
            suitability["overall"] = "poor"
        
        return suitability
    
    def _analyze_feature_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between features"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {"message": "Insufficient numeric features for correlation analysis"}
        
        corr_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": round(corr_val, 3)
                    })
        
        return {
            "high_correlations": high_corr_pairs,
            "potential_multicollinearity": len(high_corr_pairs) > 0
        }
    
    def _suggest_visualizations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Suggest appropriate visualizations"""
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            suggestions.append({
                "type": "histogram",
                "purpose": "Show distribution of numeric features",
                "columns": numeric_cols[:3]  # Limit suggestions
            })
            
            if len(numeric_cols) >= 2:
                suggestions.append({
                    "type": "scatter_plot",
                    "purpose": "Show relationships between numeric features",
                    "columns": numeric_cols[:2]
                })
                
                suggestions.append({
                    "type": "correlation_heatmap",
                    "purpose": "Visualize feature correlations",
                    "columns": numeric_cols
                })
        
        if categorical_cols:
            suggestions.append({
                "type": "bar_chart",
                "purpose": "Show categorical feature distributions",
                "columns": categorical_cols[:2]
            })
        
        if numeric_cols and categorical_cols:
            suggestions.append({
                "type": "box_plot",
                "purpose": "Compare numeric distributions across categories",
                "columns": [numeric_cols[0], categorical_cols[0]]
            })
        
        return suggestions
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall data quality score"""
        scores = {}
        
        # Completeness score (0-100)
        completeness = analysis["data_quality"]["completeness"]["completeness_ratio"]
        scores["completeness"] = round(completeness * 100, 1)
        
        # Consistency score (0-100)
        duplicate_ratio = analysis["data_quality"]["consistency"]["duplicate_ratio"]
        scores["consistency"] = round((1 - duplicate_ratio) * 100, 1)
        
        # Size adequacy score (0-100)
        size_category = analysis["shape_analysis"]["size_category"]
        size_scores = {"very_small": 20, "small": 50, "medium": 80, "large": 95, "very_large": 100}
        scores["size_adequacy"] = size_scores.get(size_category, 50)
        
        # Feature diversity score (0-100)
        feature_counts = analysis["feature_analysis"]["counts"]
        total_features = sum(feature_counts.values())
        diversity = min(100, (total_features / 5) * 100)  # 5+ features = 100%
        scores["feature_diversity"] = round(diversity, 1)
        
        # Overall score
        overall = sum(scores.values()) / len(scores)
        scores["overall"] = round(overall, 1)
        
        return scores
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary of the analysis"""
        metadata = analysis["metadata"]
        shape = analysis["shape_analysis"]
        quality = analysis["overall_quality_score"]
        
        summary_parts = [
            f"Dataset contains {metadata['rows']} rows and {metadata['columns']} columns.",
            f"Overall data quality score: {quality['overall']}/100."
        ]
        
        # Add size assessment
        size_map = {
            "very_small": "very small (consider collecting more data)",
            "small": "small (adequate for simple models)",
            "medium": "medium-sized (good for most ML tasks)",
            "large": "large (excellent for complex models)",
            "very_large": "very large (suitable for deep learning)"
        }
        summary_parts.append(f"Dataset size is {size_map[shape['size_category']]}.")
        
        # Add feature type summary
        feature_types = analysis["feature_analysis"]["counts"]
        feature_summary = []
        for ftype, count in feature_types.items():
            if count > 0:
                feature_summary.append(f"{count} {ftype}")
        
        if feature_summary:
            summary_parts.append(f"Features include: {', '.join(feature_summary)}.")
        
        # Add data quality insights
        if quality["completeness"] < 90:
            summary_parts.append("⚠️  Dataset has missing values that need attention.")
        
        if quality["consistency"] < 95:
            summary_parts.append("⚠️  Dataset contains duplicate rows.")
        
        # Add ML readiness assessment
        ml_suitability = analysis["ml_insights"]["dataset_suitability"]["overall"]
        readiness_map = {
            "excellent": "✅ Dataset is excellent for machine learning.",
            "good": "✅ Dataset is good for machine learning with minimal preprocessing.",
            "fair": "⚠️  Dataset needs some preprocessing before ML use.",
            "poor": "❌ Dataset requires significant cleanup before ML use."
        }
        summary_parts.append(readiness_map[ml_suitability])
        
        return " ".join(summary_parts)


# Main execution function for Coral Protocol integration
async def main(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the DataAnalysisAgent
    Compatible with Coral Protocol agent standards
    """
    agent = DataAnalysisAgent()
    return await agent.analyze_dataset(input_data)


# For testing and standalone usage
if __name__ == "__main__":
    import asyncio
    
    # Example usage with sample data
    sample_data = {
        "file_path": "./datasets/iris.csv"  # Update path as needed
    }
    
    async def test_agent():
        result = await main(sample_data)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_agent())