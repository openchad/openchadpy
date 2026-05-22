import sys
import os
import logging
from typing import Tuple, Optional
from pathlib import Path
import glob
logger = logging.getLogger(__name__)
# Default thresholds
MIN_FREE_VRAM_MB = 512  # Minimum free VRAM in MB to allow model loading

def check_vram(required_mb: Optional[float] = None) -> Tuple[bool, Optional[str], float]:
    """
    Check if sufficient VRAM is available.
    Args:
        required_mb: Optional minimum required VRAM in MB. If None, uses MIN_FREE_VRAM_MB.
    Returns:
        Tuple of (is_safe, error_message, free_vram_mb).
        is_safe: True if enough VRAM is available.
        error_message: None if safe, otherwise a descriptive error.
        free_vram_mb: Amount of free VRAM detected (or -1 if unknown).
    """
    threshold = required_mb if required_mb is not None else MIN_FREE_VRAM_MB
    # 1. Try NVIDIA (pynvml) - works on Windows/Linux
    try:
        free_mb = _get_nvidia_vram()
        if free_mb is not None:
            if free_mb < threshold:
                return False, f"NVIDIA GPU: Low Total VRAM ({free_mb:.0f} MB free, need {threshold:.0f} MB)", free_mb
            logger.debug(f"NVIDIA Total VRAM check passed: {free_mb:.0f} MB free")
            return True, None, free_mb
    except Exception as e:
        logger.debug(f"NVIDIA VRAM check skipped: {e}")
    # 2. Try AMD (pyrsmi) - works on Windows/Linux
    try:
        free_mb = _get_amd_rocm_vram()
        if free_mb is not None:
            if free_mb < threshold:
                return False, f"AMD GPU: Low Total VRAM ({free_mb:.0f} MB free, need {threshold:.0f} MB)", free_mb
            logger.debug(f"AMD Total VRAM check passed: {free_mb:.0f} MB free")
            return True, None, free_mb
    except Exception as e:
        logger.debug(f"AMD VRAM check skipped: {e}")
    # 3. Fallback: assume OK but return -1 to indicate unknown
    logger.warning("Could not determine VRAM usage; proceeding with load.")
    return True, None, -1

def _get_nvidia_vram() -> Optional[float]:
    """
    Get total free VRAM from all NVIDIA GPUs using pynvml.
    Returns:
        Total free VRAM in MB, or None if not available.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None
            total_free_mb = 0.0
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_free_mb += info.free / (1024 * 1024)
            return total_free_mb
        finally:
            pynvml.nvmlShutdown()
    except ImportError:
        logger.debug("pynvml not installed")
        return None
    except Exception as e:
        logger.debug(f"pynvml error: {e}")
        return None

def _get_amd_rocm_vram() -> Optional[float]:
    """Get total free VRAM using ROCm SMI Python bindings across all devices."""
    try:
        from pyrsmi import rocml
        rocml.smi_initialize()
        try:
            device_count = rocml.smi_get_device_count()
            if device_count > 0:
                total_free_mb = 0.0
                for i in range(device_count):
                    mem_info = rocml.smi_get_device_memory_total(i)
                    mem_used = rocml.smi_get_device_memory_used(i)
                    total_free_mb += (mem_info - mem_used) / (1024 * 1024)
                return total_free_mb
        finally:
            rocml.smi_shutdown()
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"ROCm SMI error: {e}")
        return None
