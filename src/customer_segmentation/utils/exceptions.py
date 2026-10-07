class CustomerSegmentationError(Exception):
    """Base exception for customer segmentation project."""


class DataValidationError(CustomerSegmentationError):
    """Raised when data validation fails."""


class DataProcessingError(CustomerSegmentationError):
    """Raised when data preprocessing fails."""


class FeatureEngineeringError(CustomerSegmentationError):
    """Raised when feature engineering fails."""


class ModelTrainingError(CustomerSegmentationError):
    """Raised when model training fails."""


class ModelEvaluationError(CustomerSegmentationError):
    """Raised when model evaluation fails."""


class ConfigurationError(CustomerSegmentationError):
    """Raised when configuration is invalid."""
