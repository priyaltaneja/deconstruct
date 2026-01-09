"""
Text Analysis Utilities
Functions for analyzing document text and images to detect complexity markers.
"""

from typing import List
import numpy as np


def detect_multicolumn(image) -> bool:
    """
    Detect multi-column layout using basic heuristics.

    Args:
        image: PIL Image object

    Returns:
        True if multi-column layout is detected
    """
    # Simplified heuristic - would use vision model in production
    # Could analyze vertical whitespace gaps
    img_array = np.array(image.convert('L'))  # Convert to grayscale

    # Check for consistent vertical gaps (columns separated by whitespace)
    width = img_array.shape[1]
    mid_section = img_array[:, width//3:2*width//3]

    # If middle section has significantly more whitespace, likely multi-column
    mean_mid = np.mean(mid_section)
    mean_overall = np.mean(img_array)

    # Higher mean = more whitespace (white = 255)
    return mean_mid > mean_overall * 1.2


def check_image_quality(image) -> bool:
    """
    Check if image is a low quality scan.

    Args:
        image: PIL Image object

    Returns:
        True if image appears to be low quality
    """
    img_array = np.array(image)
    variance = np.var(img_array)

    # Low variance suggests poor contrast / low quality
    return variance < 1000


def check_language_complexity(text: str) -> bool:
    """
    Check for ambiguous or complex legal/technical language.

    Args:
        text: Text sample to analyze

    Returns:
        True if complex language patterns are detected
    """
    complex_words = [
        # Legal terms
        "notwithstanding",
        "wherein",
        "thereof",
        "hereby",
        "aforesaid",
        "hereinafter",
        "whereas",
        "forthwith",
        "heretofore",
        # Technical/ambiguous
        "respectively",
        "aforementioned",
        "pursuant",
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in complex_words)


def detect_mixed_language(text: str) -> bool:
    """
    Detect if multiple languages are present in the text.

    Args:
        text: Text sample to analyze

    Returns:
        True if multiple languages detected
    """
    # Simple heuristic: check for non-ASCII characters mixed with ASCII
    ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    non_ascii_chars = sum(1 for c in text if ord(c) >= 128 and c.isalpha())

    total_alpha = ascii_chars + non_ascii_chars
    if total_alpha == 0:
        return False

    # If significant portion is non-ASCII, likely mixed language
    return non_ascii_chars > total_alpha * 0.1


def estimate_entity_count(text: str) -> int:
    """
    Rough estimate of extractable entities in text.

    Args:
        text: Text sample to analyze

    Returns:
        Estimated count of named entities
    """
    # Count capitalized words as potential entities
    words = text.split()
    entity_count = sum(1 for w in words if w and w[0].isupper() and len(w) > 1)
    return entity_count


def classify_document_type(text: str) -> str:
    """
    Classify document type from text sample.

    Args:
        text: Text sample to analyze

    Returns:
        Document type: "legal", "financial", "technical", or "general"
    """
    text_lower = text.lower()

    # Legal document indicators
    legal_keywords = [
        "agreement",
        "clause",
        "party",
        "whereas",
        "hereby",
        "contract",
        "obligation",
        "liability",
        "indemnify",
        "termination",
        "jurisdiction",
    ]

    # Financial document indicators
    financial_keywords = [
        "revenue",
        "balance sheet",
        "fiscal",
        "quarterly",
        "annual report",
        "earnings",
        "assets",
        "liabilities",
        "cash flow",
        "income statement",
    ]

    # Technical document indicators
    technical_keywords = [
        "api",
        "specification",
        "requirement",
        "function",
        "architecture",
        "implementation",
        "interface",
        "endpoint",
        "database",
        "schema",
    ]

    # Count matches
    legal_count = sum(1 for kw in legal_keywords if kw in text_lower)
    financial_count = sum(1 for kw in financial_keywords if kw in text_lower)
    technical_count = sum(1 for kw in technical_keywords if kw in text_lower)

    # Return type with highest match count
    max_count = max(legal_count, financial_count, technical_count)

    if max_count == 0:
        return "general"

    if legal_count == max_count:
        return "legal"
    elif financial_count == max_count:
        return "financial"
    elif technical_count == max_count:
        return "technical"

    return "general"


def extract_text_from_images(images: List, max_chars: int = 1000) -> str:
    """
    Extract text from images using OCR.

    Args:
        images: List of PIL Image objects
        max_chars: Maximum characters to extract per image

    Returns:
        Combined text from all images
    """
    import pytesseract

    text_sample = ""
    for img in images:
        text_sample += pytesseract.image_to_string(img)[:max_chars]

    return text_sample
