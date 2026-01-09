"""
Cost Calculation Utilities
Calculates API and compute costs for extraction operations.
"""

from typing import Optional, Dict, Any

# Model pricing (per token)
MODEL_COSTS = {
    "claude-opus-4-5": {
        "input": 15.0 / 1_000_000,
        "output": 75.0 / 1_000_000,
    },
    "claude-opus-4-5-20251101": {
        "input": 15.0 / 1_000_000,
        "output": 75.0 / 1_000_000,
    },
    "claude-sonnet-4": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
    },
    "llama-3.2-vision": {
        "input": 0.0001 / 1000,  # Approximate GPU cost
        "output": 0.0001 / 1000,
    },
    "llama-3.2-11b-vision": {
        "input": 0.0001 / 1000,
        "output": 0.0001 / 1000,
    },
}


def calculate_cost(usage: Any, model_name: str) -> float:
    """
    Calculate API cost based on token usage.

    Args:
        usage: Usage object with input_tokens and output_tokens attributes
        model_name: Name of the model used

    Returns:
        Estimated cost in USD
    """
    # Normalize model name
    model_key = model_name.lower()

    # Try exact match first
    if model_key in MODEL_COSTS:
        pricing = MODEL_COSTS[model_key]
    else:
        # Try partial match
        for key in MODEL_COSTS:
            if key in model_key or model_key in key:
                pricing = MODEL_COSTS[key]
                break
        else:
            # Default to zero cost if model not found
            return 0.0

    input_cost = getattr(usage, 'input_tokens', 0) * pricing["input"]
    output_cost = getattr(usage, 'output_tokens', 0) * pricing["output"]

    return input_cost + output_cost


def estimate_gpu_cost(processing_time_ms: float, gpu_type: str = "A10G") -> float:
    """
    Estimate GPU compute cost based on processing time.

    Args:
        processing_time_ms: Processing time in milliseconds
        gpu_type: Type of GPU used

    Returns:
        Estimated cost in USD
    """
    # Modal GPU pricing (approximate, per second)
    gpu_rates = {
        "A10G": 0.000306,  # ~$1.10/hour
        "A100": 0.001528,  # ~$5.50/hour
        "T4": 0.000164,    # ~$0.59/hour
    }

    rate = gpu_rates.get(gpu_type, gpu_rates["A10G"])
    processing_time_s = processing_time_ms / 1000

    return processing_time_s * rate


def get_model_pricing(model_name: str) -> Optional[Dict[str, float]]:
    """
    Get pricing information for a model.

    Args:
        model_name: Name of the model

    Returns:
        Dict with 'input' and 'output' rates, or None if not found
    """
    model_key = model_name.lower()

    if model_key in MODEL_COSTS:
        return MODEL_COSTS[model_key]

    for key in MODEL_COSTS:
        if key in model_key or model_key in key:
            return MODEL_COSTS[key]

    return None
