"""
Centralized Configuration for Deconstruct GPU Backend
All environment-dependent values and constants should be defined here.
"""

import os
from typing import List

# =============================================================================
# CORS Configuration
# =============================================================================

def get_cors_origins() -> List[str]:
    """
    Get allowed CORS origins from environment.
    In production, set CORS_ORIGINS to comma-separated list of allowed domains.
    """
    env_origins = os.getenv("CORS_ORIGINS")
    if env_origins:
        return [origin.strip() for origin in env_origins.split(",")]

    # Default development origins
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


# =============================================================================
# Extraction Configuration
# =============================================================================

# Default complexity threshold for System 1/System 2 routing
# Documents with complexity score >= threshold use System 2 (Vision LLM)
DEFAULT_COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.7"))

# API server port
API_PORT = int(os.getenv("API_PORT", "8000"))


# =============================================================================
# Model Configuration - All Local on Modal
# =============================================================================

# System 1: PaddleOCR + Text LLM (for simple documents)
# Uses OCR to extract text, then sends to a fast text-only LLM
SYSTEM1_TEXT_MODEL = os.getenv("SYSTEM1_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SYSTEM1_MAX_TOKENS = int(os.getenv("SYSTEM1_MAX_TOKENS", "4096"))
SYSTEM1_TEMPERATURE = float(os.getenv("SYSTEM1_TEMPERATURE", "0.1"))

# System 2: Vision LLM (for complex documents with tables/layouts)
# Sends images directly to a vision-language model
SYSTEM2_VISION_MODEL = os.getenv("SYSTEM2_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
SYSTEM2_MAX_TOKENS = int(os.getenv("SYSTEM2_MAX_TOKENS", "8192"))
SYSTEM2_TEMPERATURE = float(os.getenv("SYSTEM2_TEMPERATURE", "0.1"))

# Image processing
SYSTEM1_DPI = int(os.getenv("SYSTEM1_DPI", "200"))
SYSTEM2_DPI = int(os.getenv("SYSTEM2_DPI", "300"))
MAX_PAGES_FOR_EXTRACTION = int(os.getenv("MAX_PAGES_FOR_EXTRACTION", "10"))

# Scan settings
SCAN_FIRST_N_PAGES = int(os.getenv("SCAN_FIRST_N_PAGES", "3"))
SCAN_TEXT_SAMPLE_SIZE = int(os.getenv("SCAN_TEXT_SAMPLE_SIZE", "2000"))


# =============================================================================
# GPU Configuration
# =============================================================================

# System 1 GPU (PaddleOCR + Text LLM)
SYSTEM1_GPU = os.getenv("SYSTEM1_GPU", "A10G")

# System 2 GPU (Vision LLM - needs more VRAM)
SYSTEM2_GPU = os.getenv("SYSTEM2_GPU", "A10G")


# =============================================================================
# Cost Estimation (GPU time based)
# =============================================================================

# Approximate GPU costs per second
GPU_COSTS_PER_SECOND = {
    "T4": 0.000164,      # ~$0.59/hr
    "A10G": 0.000306,    # ~$1.10/hr
    "A100": 0.001528,    # ~$5.50/hr
    "A100-80GB": 0.002778,  # ~$10/hr
}


def estimate_gpu_cost(processing_time_ms: float, gpu_type: str) -> float:
    """Estimate cost based on GPU time."""
    seconds = processing_time_ms / 1000
    rate = GPU_COSTS_PER_SECOND.get(gpu_type, GPU_COSTS_PER_SECOND["A10G"])
    return seconds * rate


# =============================================================================
# Timeouts
# =============================================================================

# Modal function timeouts (in seconds)
SYSTEM1_TIMEOUT = int(os.getenv("SYSTEM1_TIMEOUT", "300"))
SYSTEM2_TIMEOUT = int(os.getenv("SYSTEM2_TIMEOUT", "600"))
BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "1800"))

# Subprocess timeout for local API (in seconds)
SUBPROCESS_TIMEOUT = int(os.getenv("SUBPROCESS_TIMEOUT", "900"))


# =============================================================================
# File Limits
# =============================================================================

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
