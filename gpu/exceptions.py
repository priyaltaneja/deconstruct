"""
Custom Exception Hierarchy for Deconstruct
Provides specific exception types for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class DeconstructError(Exception):
    """Base exception for all Deconstruct errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        base = self.message
        if self.details:
            base += f" | Details: {self.details}"
        if self.cause:
            base += f" | Caused by: {type(self.cause).__name__}: {self.cause}"
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "details": self.details,
        }


class ExtractionError(DeconstructError):
    """Error during document extraction process."""

    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        extraction_tier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if document_id:
            details["document_id"] = document_id
        if extraction_tier:
            details["extraction_tier"] = extraction_tier
        super().__init__(message, details, cause)
        self.document_id = document_id
        self.extraction_tier = extraction_tier


class ValidationError(DeconstructError):
    """Error during data validation (schema, format, etc.)."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if field:
            details["field"] = field
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)[:100]  # Truncate for safety
        super().__init__(message, details, cause)


class DatabaseError(DeconstructError):
    """Error during database operations."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if operation:
            details["operation"] = operation
        if table:
            details["table"] = table
        super().__init__(message, details, cause)


class ModalError(DeconstructError):
    """Error during Modal function execution."""

    def __init__(
        self,
        message: str,
        function_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if function_name:
            details["function_name"] = function_name
        super().__init__(message, details, cause)


class PDFProcessingError(DeconstructError):
    """Error during PDF processing (conversion, OCR, etc.)."""

    def __init__(
        self,
        message: str,
        file_name: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if file_name:
            details["file_name"] = file_name
        if operation:
            details["operation"] = operation
        super().__init__(message, details, cause)


class LLMError(DeconstructError):
    """Error during LLM API calls."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if model:
            details["model"] = model
        if provider:
            details["provider"] = provider
        super().__init__(message, details, cause)


class ConfigurationError(DeconstructError):
    """Error in configuration (missing env vars, invalid settings)."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details, cause)


class RateLimitError(DeconstructError):
    """Error when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, details, cause)


class TimeoutError(DeconstructError):
    """Error when operation times out."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[int] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        super().__init__(message, details, cause)


# =============================================================================
# Error Message Mapping (for sanitized API responses)
# =============================================================================

# User-friendly error messages for each exception type
ERROR_MESSAGES = {
    ExtractionError: "Document extraction failed. Please try again.",
    ValidationError: "Invalid document format or data.",
    DatabaseError: "Failed to save results. Please try again.",
    ModalError: "Processing service temporarily unavailable.",
    PDFProcessingError: "Failed to process PDF. Please check the file format.",
    LLMError: "AI model temporarily unavailable. Please try again.",
    ConfigurationError: "Service configuration error. Please contact support.",
    RateLimitError: "Too many requests. Please wait and try again.",
    TimeoutError: "Request timed out. Please try again with a smaller document.",
}


def get_user_friendly_message(error: Exception) -> str:
    """Get a user-friendly error message for the given exception."""
    for error_type, message in ERROR_MESSAGES.items():
        if isinstance(error, error_type):
            return message
    return "An unexpected error occurred. Please try again."


def sanitize_error_for_response(error: Exception) -> Dict[str, Any]:
    """
    Convert an exception to a sanitized API response.
    Does not expose internal details to the client.
    """
    if isinstance(error, DeconstructError):
        return {
            "error": get_user_friendly_message(error),
            "error_type": type(error).__name__,
        }
    return {
        "error": "An unexpected error occurred. Please try again.",
        "error_type": "UnexpectedError",
    }
