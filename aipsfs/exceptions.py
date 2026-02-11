# exceptions.py


class StockAnalysisError(Exception):
    """Base exception for stock analysis errors."""

    pass


class DataFetchError(StockAnalysisError):
    """Exception for data fetching errors."""

    pass


class ModelTrainingError(StockAnalysisError):
    """Exception for model training errors."""

    pass
