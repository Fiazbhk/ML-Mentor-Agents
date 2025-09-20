"""
ModelSelectionAgent - Intelligent ML model recommendation system
Part of the ML Mentor system for making machine learning accessible to beginners
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProblemType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


class DatasetSize(Enum):
    VERY_SMALL = "very_small"  # < 100 samples
    SMALL = "small"  # 100-1000 samples
    MEDIUM = "medium"  # 1000-10000 samples
    LARGE = "large"  # 10000-100000 samples
    VERY_LARGE = "very_large"  # > 100000 samples


class ComplexityLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class ModelRecommendation:
    """Data class representing a model recommendation"""
    name: str
    algorithm_family: str
    confidence_score: float
    complexity_level: ComplexityLevel
    training_time: str
    interpretability: str
    pros: List[str]
    cons: List[str]
    use_cases: List[str]
    sklearn_class: str
    hyperparameters: Dict[str, Any]
    performance_expectation: str
    beginner_explanation: str


class ModelSelectionAgent:
    """
    Intelligent agent that recommends the best ML models based on dataset characteristics,
    problem type, and user skill level. Provides detailed explanations for beginners.
    """
    
    def __init__(self):
        self.name = "ModelSelectionAgent"
        self.version = "1.0.0"
        self.description = "Intelligent ML model recommendation system"
        self.model_database = self._initialize_model_database()
        
    async def recommend_models(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main recommendation method that analyzes input and suggests best models
        
        Args:
            input_data: Dictionary containing dataset analysis and user preferences
            
        Returns:
            Model recommendations with explanations
        """
        try:
            logger.info(f"Starting model selection at {datetime.now()}")
            
            # Parse input data
            analysis_results = self._parse_input_data(input_data)
            
            if not analysis_results:
                return {"error": "Invalid or insufficient input data"}
            
            # Determine problem type
            problem_type = self._determine_problem_type(analysis_results)
            
            # Assess dataset characteristics
            dataset_profile = self._create_dataset_profile(analysis_results)
            
            # Get user preferences
            user_preferences = self._extract_user_preferences(input_data)
            
            # Generate model recommendations
            recommendations = self._generate_recommendations(
                problem_type, dataset_profile, user_preferences
            )
            
            # Rank and filter recommendations
            top_recommendations = self._rank_recommendations(
                recommendations, dataset_profile, user_preferences
            )
            
            # Generate explanations and guidance
            result = {
                "success": True,
                "problem_type": problem_type.value,
                "dataset_profile": asdict(dataset_profile),
                "recommendations": [asdict(rec) for rec in top_recommendations],
                "summary": self._generate_summary(problem_type, top_recommendations),
                "next_steps": self._generate_next_steps(top_recommendations[0] if top_recommendations else None),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Model selection completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during model selection: {str(e)}")
            return {"error": f"Model selection failed: {str(e)}"}
    
    def _parse_input_data(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse and validate input data from DataAnalysisAgent or direct input"""
        
        # Handle input from DataAnalysisAgent
        if "analysis" in input_data:
            return input_data["analysis"]
        
        # Handle direct dataset characteristics input
        if all(key in input_data for key in ["rows", "columns", "target_type"]):
            return {
                "metadata": {"rows": input_data["rows"], "columns": input_data["columns"]},
                "feature_analysis": {"target_type": input_data["target_type"]},
                "data_quality": input_data.get("data_quality", {}),
                "ml_insights": input_data.get("ml_insights", {})
            }
        
        return None
    
    def _determine_problem_type(self, analysis: Dict[str, Any]) -> ProblemType:
        """Determine the ML problem type based on dataset analysis"""
        
        # Check for explicit problem type in ML insights
        if "ml_insights" in analysis and "problem_type" in analysis["ml_insights"]:
            problem_hints = analysis["ml_insights"]["problem_type"]
            if isinstance(problem_hints, list) and problem_hints:
                first_hint = problem_hints[0].lower()
                
                if "binary classification" in first_hint:
                    return ProblemType.BINARY_CLASSIFICATION
                elif "multi-class classification" in first_hint or "classification" in first_hint:
                    return ProblemType.MULTICLASS_CLASSIFICATION
                elif "regression" in first_hint:
                    return ProblemType.REGRESSION
        
        # Fallback analysis based on feature characteristics
        feature_analysis = analysis.get("feature_analysis", {})
        metadata = analysis.get("metadata", {})
        
        # Look for target variable hints
        if "target_type" in feature_analysis:
            target_type = feature_analysis["target_type"]
            if target_type == "binary":
                return ProblemType.BINARY_CLASSIFICATION
            elif target_type == "categorical":
                return ProblemType.MULTICLASS_CLASSIFICATION
            elif target_type == "numeric":
                return ProblemType.REGRESSION
        
        # Default based on dataset size and structure
        num_rows = metadata.get("rows", 0)
        if num_rows < 50:
            return ProblemType.CLUSTERING  # Assume unsupervised for very small datasets
        else:
            return ProblemType.MULTICLASS_CLASSIFICATION  # Most common case
    
    @dataclass
    class DatasetProfile:
        """Profile of dataset characteristics for model selection"""
        size_category: DatasetSize
        num_features: int
        feature_types: Dict[str, int]
        data_quality_score: float
        has_missing_values: bool
        has_categorical_features: bool
        has_numeric_features: bool
        is_imbalanced: bool
        noise_level: str
        complexity_indicators: List[str]
    
    def _create_dataset_profile(self, analysis: Dict[str, Any]) -> 'ModelSelectionAgent.DatasetProfile':
        """Create a comprehensive dataset profile"""
        
        metadata = analysis.get("metadata", {})
        feature_analysis = analysis.get("feature_analysis", {})
        data_quality = analysis.get("data_quality", {})
        
        num_rows = metadata.get("rows", 0)
        num_cols = metadata.get("columns", 0)
        
        # Determine size category
        if num_rows < 100:
            size_category = DatasetSize.VERY_SMALL
        elif num_rows < 1000:
            size_category = DatasetSize.SMALL
        elif num_rows < 10000:
            size_category = DatasetSize.MEDIUM
        elif num_rows < 100000:
            size_category = DatasetSize.LARGE
        else:
            size_category = DatasetSize.VERY_LARGE
        
        # Extract feature type counts
        feature_counts = feature_analysis.get("counts", {})
        feature_types = {
            "numeric": feature_counts.get("numeric", 0),
            "categorical": feature_counts.get("categorical", 0),
            "datetime": feature_counts.get("datetime", 0),
            "boolean": feature_counts.get("boolean", 0)
        }
        
        # Data quality assessment
        quality_score = 85.0  # Default
        if "completeness" in data_quality:
            completeness = data_quality["completeness"].get("completeness_ratio", 1.0)
            quality_score = completeness * 100
        
        has_missing = False
        if "completeness" in data_quality:
            missing_cells = data_quality["completeness"].get("missing_cells", 0)
            has_missing = missing_cells > 0
        
        # Complexity indicators
        complexity_indicators = []
        if num_cols > 50:
            complexity_indicators.append("high_dimensionality")
        if has_missing:
            complexity_indicators.append("missing_data")
        if feature_types["categorical"] > feature_types["numeric"]:
            complexity_indicators.append("categorical_heavy")
        
        return self.DatasetProfile(
            size_category=size_category,
            num_features=num_cols,
            feature_types=feature_types,
            data_quality_score=quality_score,
            has_missing_values=has_missing,
            has_categorical_features=feature_types["categorical"] > 0,
            has_numeric_features=feature_types["numeric"] > 0,
            is_imbalanced=False,  # Would need more analysis to determine
            noise_level="medium",  # Default assumption
            complexity_indicators=complexity_indicators
        )
    
    def _extract_user_preferences(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user preferences and constraints"""
        preferences = input_data.get("preferences", {})
        
        return {
            "skill_level": preferences.get("skill_level", "beginner"),
            "interpretability_importance": preferences.get("interpretability", "high"),
            "training_time_preference": preferences.get("training_time", "fast"),
            "accuracy_vs_speed": preferences.get("accuracy_vs_speed", "balanced"),
            "deployment_constraints": preferences.get("deployment", {}),
            "preferred_libraries": preferences.get("libraries", ["scikit-learn"])
        }
    
    def _initialize_model_database(self) -> Dict[str, List[ModelRecommendation]]:
        """Initialize the comprehensive model database"""
        
        models = {
            ProblemType.BINARY_CLASSIFICATION: [
                ModelRecommendation(
                    name="Logistic Regression",
                    algorithm_family="Linear Models",
                    confidence_score=0.9,
                    complexity_level=ComplexityLevel.BEGINNER,
                    training_time="Very Fast",
                    interpretability="High",
                    pros=[
                        "Easy to understand and interpret",
                        "Fast training and prediction",
                        "No hyperparameter tuning needed",
                        "Works well with small datasets",
                        "Provides probability estimates"
                    ],
                    cons=[
                        "Assumes linear relationship",
                        "Can struggle with complex patterns",
                        "Sensitive to outliers",
                        "Requires feature scaling"
                    ],
                    use_cases=[
                        "Email spam detection",
                        "Medical diagnosis (yes/no)",
                        "Marketing response prediction",
                        "Credit approval"
                    ],
                    sklearn_class="LogisticRegression",
                    hyperparameters={
                        "C": 1.0,
                        "random_state": 42,
                        "max_iter": 1000
                    },
                    performance_expectation="Good baseline performance, 70-85% accuracy on most problems",
                    beginner_explanation="Think of logistic regression as drawing a line to separate two groups. It's like asking 'which side of the line does this point fall on?' to make predictions."
                ),
                
                ModelRecommendation(
                    name="Random Forest",
                    algorithm_family="Ensemble Methods",
                    confidence_score=0.95,
                    complexity_level=ComplexityLevel.BEGINNER,
                    training_time="Medium",
                    interpretability="Medium",
                    pros=[
                        "Excellent out-of-the-box performance",
                        "Handles mixed data types well",
                        "Resistant to overfitting",
                        "Provides feature importance",
                        "No feature scaling required"
                    ],
                    cons=[
                        "Can be slow on large datasets",
                        "Less interpretable than single trees",
                        "May overfit on very noisy data",
                        "Memory intensive"
                    ],
                    use_cases=[
                        "Customer churn prediction",
                        "Fraud detection",
                        "Image classification",
                        "Risk assessment"
                    ],
                    sklearn_class="RandomForestClassifier",
                    hyperparameters={
                        "n_estimators": 100,
                        "random_state": 42,
                        "max_depth": None
                    },
                    performance_expectation="Very good performance, typically 80-90% accuracy",
                    beginner_explanation="Random Forest is like asking 100 different experts for their opinion and taking a vote. Each expert looks at the data slightly differently, making the final decision more reliable."
                ),
                
                ModelRecommendation(
                    name="Support Vector Machine (SVM)",
                    algorithm_family="Kernel Methods",
                    confidence_score=0.85,
                    complexity_level=ComplexityLevel.INTERMEDIATE,
                    training_time="Medium",
                    interpretability="Low",
                    pros=[
                        "Works well with high-dimensional data",
                        "Effective for small datasets",
                        "Memory efficient",
                        "Versatile with different kernels"
                    ],
                    cons=[
                        "Slow on large datasets",
                        "Requires feature scaling",
                        "No probability estimates by default",
                        "Sensitive to feature selection"
                    ],
                    use_cases=[
                        "Text classification",
                        "Gene classification",
                        "Image recognition",
                        "High-dimensional problems"
                    ],
                    sklearn_class="SVC",
                    hyperparameters={
                        "C": 1.0,
                        "kernel": "rbf",
                        "random_state": 42,
                        "probability": True
                    },
                    performance_expectation="Good performance on small to medium datasets, 75-88% accuracy",
                    beginner_explanation="SVM finds the best boundary between classes by creating a 'margin' - like finding the widest possible road between two neighborhoods."
                )
            ],
            
            ProblemType.MULTICLASS_CLASSIFICATION: [
                ModelRecommendation(
                    name="Random Forest",
                    algorithm_family="Ensemble Methods",
                    confidence_score=0.95,
                    complexity_level=ComplexityLevel.BEGINNER,
                    training_time="Medium",
                    interpretability="Medium",
                    pros=[
                        "Handles multiple classes naturally",
                        "Excellent default performance",
                        "Works with mixed data types",
                        "Provides feature importance",
                        "Resistant to overfitting"
                    ],
                    cons=[
                        "Can be memory intensive",
                        "Slower than linear models",
                        "May struggle with very high cardinality",
                        "Less interpretable than decision trees"
                    ],
                    use_cases=[
                        "Species classification",
                        "Product categorization",
                        "Document classification",
                        "Image recognition"
                    ],
                    sklearn_class="RandomForestClassifier",
                    hyperparameters={
                        "n_estimators": 100,
                        "random_state": 42,
                        "max_depth": None
                    },
                    performance_expectation="Very good multi-class performance, typically 80-90% accuracy",
                    beginner_explanation="For multiple categories, Random Forest asks each of its 'expert trees' which category they think it is, then takes a vote across all categories."
                ),
                
                ModelRecommendation(
                    name="Gradient Boosting",
                    algorithm_family="Boosting",
                    confidence_score=0.90,
                    complexity_level=ComplexityLevel.INTERMEDIATE,
                    training_time="Slow",
                    interpretability="Medium",
                    pros=[
                        "Often achieves highest accuracy",
                        "Handles complex patterns well",
                        "Good with mixed data types",
                        "Provides feature importance"
                    ],
                    cons=[
                        "Prone to overfitting",
                        "Requires careful tuning",
                        "Slower to train",
                        "More complex to understand"
                    ],
                    use_cases=[
                        "Competition problems",
                        "Complex classification tasks",
                        "When accuracy is paramount",
                        "Structured data problems"
                    ],
                    sklearn_class="GradientBoostingClassifier",
                    hyperparameters={
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "random_state": 42
                    },
                    performance_expectation="Excellent performance with tuning, 85-95% accuracy",
                    beginner_explanation="Gradient Boosting learns from its mistakes - each new model tries to fix the errors of the previous ones, like a student learning from feedback."
                ),
                
                ModelRecommendation(
                    name="K-Nearest Neighbors (KNN)",
                    algorithm_family="Instance-Based",
                    confidence_score=0.80,
                    complexity_level=ComplexityLevel.BEGINNER,
                    training_time="Very Fast",
                    interpretability="High",
                    pros=[
                        "Very easy to understand",
                        "No training period needed",
                        "Works well with small datasets",
                        "Naturally handles multiple classes",
                        "Can capture complex decision boundaries"
                    ],
                    cons=[
                        "Slow predictions on large datasets",
                        "Sensitive to irrelevant features",
                        "Requires feature scaling",
                        "Can be affected by local noise"
                    ],
                    use_cases=[
                        "Recommendation systems",
                        "Pattern recognition",
                        "Outlier detection",
                        "Small dataset problems"
                    ],
                    sklearn_class="KNeighborsClassifier",
                    hyperparameters={
                        "n_neighbors": 5,
                        "weights": "uniform"
                    },
                    performance_expectation="Good performance on small to medium datasets, 70-85% accuracy",
                    beginner_explanation="KNN is like asking 'what are the 5 most similar examples I've seen before, and what category were they?' It's like getting advice from your most similar neighbors."
                )
            ],
            
            ProblemType.REGRESSION: [
                ModelRecommendation(
                    name="Linear Regression",
                    algorithm_family="Linear Models",
                    confidence_score=0.85,
                    complexity_level=ComplexityLevel.BEGINNER,
                    training_time="Very Fast",
                    interpretability="High",
                    pros=[
                        "Extremely easy to interpret",
                        "Fast training and prediction",
                        "Works well with linear relationships",
                        "No hyperparameters to tune",
                        "Provides confidence intervals"
                    ],
                    cons=[
                        "Assumes linear relationships",
                        "Sensitive to outliers",
                        "Can overfit with many features",
                        "Poor with non-linear patterns"
                    ],
                    use_cases=[
                        "Sales forecasting",
                        "Price prediction",
                        "Economic modeling",
                        "Simple trend analysis"
                    ],
                    sklearn_class="LinearRegression",
                    hyperparameters={},
                    performance_expectation="Good for linear relationships, R² typically 0.6-0.8",
                    beginner_explanation="Linear regression finds the best straight line through your data points - like drawing a trend line that minimizes the distance to all points."
                ),
                
                ModelRecommendation(
                    name="Random Forest Regressor",
                    algorithm_family="Ensemble Methods",
                    confidence_score=0.95,
                    complexity_level=ComplexityLevel.BEGINNER,
                    training_time="Medium",
                    interpretability="Medium",
                    pros=[
                        "Excellent out-of-the-box performance",
                        "Handles non-linear relationships",
                        "Works with mixed data types",
                        "Provides feature importance",
                        "Resistant to overfitting"
                    ],
                    cons=[
                        "Can be memory intensive",
                        "May not extrapolate well",
                        "Less interpretable than linear models",
                        "Can struggle with linear relationships"
                    ],
                    use_cases=[
                        "House price prediction",
                        "Stock price forecasting",
                        "Demand forecasting",
                        "Performance prediction"
                    ],
                    sklearn_class="RandomForestRegressor",
                    hyperparameters={
                        "n_estimators": 100,
                        "random_state": 42,
                        "max_depth": None
                    },
                    performance_expectation="Very good performance on most regression tasks, R² typically 0.75-0.9",
                    beginner_explanation="Random Forest for regression averages the predictions of many decision trees, like getting price estimates from multiple real estate agents and taking the average."
                ),
                
                ModelRecommendation(
                    name="Support Vector Regression (SVR)",
                    algorithm_family="Kernel Methods",
                    confidence_score=0.80,
                    complexity_level=ComplexityLevel.INTERMEDIATE,
                    training_time="Medium",
                    interpretability="Low",
                    pros=[
                        "Works well with high-dimensional data",
                        "Effective for complex patterns",
                        "Memory efficient",
                        "Good generalization"
                    ],
                    cons=[
                        "Requires feature scaling",
                        "Sensitive to hyperparameters",
                        "Slower on large datasets",
                        "Difficult to interpret"
                    ],
                    use_cases=[
                        "Financial forecasting",
                        "Engineering applications",
                        "High-dimensional regression",
                        "Non-linear pattern recognition"
                    ],
                    sklearn_class="SVR",
                    hyperparameters={
                        "C": 1.0,
                        "kernel": "rbf",
                        "gamma": "scale"
                    },
                    performance_expectation="Good performance with proper tuning, R² typically 0.7-0.85",
                    beginner_explanation="SVR creates a 'tube' around the best-fit line and only worries about points outside this tube, making it robust to outliers."
                )
            ]
        }
        
        return models
    
    def _generate_recommendations(
        self, 
        problem_type: ProblemType, 
        dataset_profile: 'ModelSelectionAgent.DatasetProfile',
        user_preferences: Dict[str, Any]
    ) -> List[ModelRecommendation]:
        """Generate model recommendations based on problem type and dataset characteristics"""
        
        # Get base models for problem type
        base_models = self.model_database.get(problem_type, [])
        
        # Score each model based on dataset characteristics
        scored_models = []
        for model in base_models:
            score = self._score_model_fit(model, dataset_profile, user_preferences)
            # Create a copy with updated confidence score
            updated_model = ModelRecommendation(
                name=model.name,
                algorithm_family=model.algorithm_family,
                confidence_score=score,
                complexity_level=model.complexity_level,
                training_time=model.training_time,
                interpretability=model.interpretability,
                pros=model.pros,
                cons=model.cons,
                use_cases=model.use_cases,
                sklearn_class=model.sklearn_class,
                hyperparameters=model.hyperparameters,
                performance_expectation=model.performance_expectation,
                beginner_explanation=model.beginner_explanation
            )
            scored_models.append(updated_model)
        
        return scored_models
    
    def _score_model_fit(
        self, 
        model: ModelRecommendation, 
        dataset_profile: 'ModelSelectionAgent.DatasetProfile',
        user_preferences: Dict[str, Any]
    ) -> float:
        """Score how well a model fits the dataset and user preferences"""
        
        base_score = model.confidence_score
        adjustments = 0
        
        # Dataset size adjustments
        if dataset_profile.size_category == DatasetSize.VERY_SMALL:
            if model.name in ["K-Nearest Neighbors (KNN)", "Logistic Regression"]:
                adjustments += 0.1
            elif model.name in ["Random Forest", "Gradient Boosting"]:
                adjustments -= 0.1
        
        elif dataset_profile.size_category == DatasetSize.VERY_LARGE:
            if model.name in ["Support Vector Machine (SVM)", "Support Vector Regression (SVR)"]:
                adjustments -= 0.2
            elif model.name in ["Linear Regression", "Logistic Regression"]:
                adjustments += 0.1
        
        # Data quality adjustments
        if dataset_profile.data_quality_score < 70:
            if model.name in ["Random Forest", "Random Forest Regressor"]:
                adjustments += 0.1  # More robust to missing data
        
        # User skill level adjustments
        skill_level = user_preferences.get("skill_level", "beginner")
        if skill_level == "beginner":
            if model.complexity_level == ComplexityLevel.BEGINNER:
                adjustments += 0.1
            elif model.complexity_level == ComplexityLevel.ADVANCED:
                adjustments -= 0.1
        
        # Interpretability preferences
        interpretability_importance = user_preferences.get("interpretability_importance", "high")
        if interpretability_importance == "high":
            if model.interpretability == "High":
                adjustments += 0.1
            elif model.interpretability == "Low":
                adjustments -= 0.1
        
        # Training time preferences
        training_time_preference = user_preferences.get("training_time_preference", "fast")
        if training_time_preference == "fast":
            if model.training_time in ["Very Fast", "Fast"]:
                adjustments += 0.05
            elif model.training_time == "Slow":
                adjustments -= 0.1
        
        final_score = max(0.1, min(1.0, base_score + adjustments))
        return round(final_score, 3)
    
    def _rank_recommendations(
        self, 
        recommendations: List[ModelRecommendation],
        dataset_profile: 'ModelSelectionAgent.DatasetProfile',
        user_preferences: Dict[str, Any]
    ) -> List[ModelRecommendation]:
        """Rank and filter recommendations"""
        
        # Sort by confidence score
        sorted_recommendations = sorted(recommendations, key=lambda x: x.confidence_score, reverse=True)
        
        # Return top 3 recommendations
        return sorted_recommendations[:3]
    
    def _generate_summary(self, problem_type: ProblemType, recommendations: List[ModelRecommendation]) -> str:
        """Generate human-readable summary"""
        
        if not recommendations:
            return "No suitable models found for your dataset."
        
        top_model = recommendations[0]
        problem_desc = {
            ProblemType.BINARY_CLASSIFICATION: "binary classification (yes/no predictions)",
            ProblemType.MULTICLASS_CLASSIFICATION: "multi-class classification (category predictions)",
            ProblemType.REGRESSION: "regression (numeric predictions)",
            ProblemType.CLUSTERING: "clustering (grouping similar data)",
        }.get(problem_type, "machine learning")
        
        summary = f"For your {problem_desc} problem, I recommend starting with **{top_model.name}** "
        summary += f"(confidence: {top_model.confidence_score:.1%}). "
        summary += f"This model is {top_model.complexity_level.value}-friendly and offers {top_model.interpretability.lower()} interpretability. "
        summary += f"{top_model.beginner_explanation}"
        
        if len(recommendations) > 1:
            summary += f" Also consider **{recommendations[1].name}** as an alternative."
        
        return summary
    
    def _generate_next_steps(self, top_model: Optional[ModelRecommendation]) -> List[str]:
        """Generate actionable next steps for the user"""
        
        if not top_model:
            return ["Collect more data and try again"]
        
        steps = [
            f"Import {top_model.sklearn_class} from scikit-learn",
            "Prepare your data (handle missing values, encode categories)",
            f"Initialize {top_model.name} with recommended parameters",
            "Split your data into training and testing sets",
            "Train the model on your training data",
            "Evaluate performance on test data",
            "Fine-tune hyperparameters if needed"
        ]
        
        # Add model-specific steps
        if "scaling" in [con.lower() for con in top_model.cons]:
            steps.insert(2, "Scale your numeric features (StandardScaler recommended)")
        
        if top_model.training_time == "Slow":
            steps.append("Consider using a smaller sample for initial experiments")
        
        return steps


# Main execution function for Coral Protocol integration
async def main(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the ModelSelectionAgent
    Compatible with Coral Protocol agent standards
    """
    agent = ModelSelectionAgent()
    return await agent.recommend_models(input_data)


# For testing and standalone usage
if __name__ == "__main__":
    import asyncio
    
    # Example usage with sample analysis data
    sample_data = {
        "analysis": {
            "metadata": {"rows": 150, "columns": 5},
            "feature_analysis": {
                "types": {
                    "numeric": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                    "categorical": ["species"]
                },
                "counts": {"numeric": 4, "categorical": 1}
            },
            "data_quality": {
                "completeness": {"completeness_ratio": 1.0, "missing_cells": 0}
            },
            "ml_insights": {
                "problem_type": ["Multi-class classification (target: species)"]
            }
        },
        "preferences": {
            "skill_level": "beginner",
            "interpretability": "high",
            "training_time": "fast"
        }
    }
    
    async def test_agent():
        result = await main(sample_data)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_agent())