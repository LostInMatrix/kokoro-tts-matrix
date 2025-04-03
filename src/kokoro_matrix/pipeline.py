import os
import sys
import torch
from kokoro import KPipeline

def initialize_pipeline(lang_code: str = 'a') -> KPipeline:
    """
    Initialize a KPipeline instance with GPU if available.

    :param lang_code: Language code (default is 'a')
    :return: An initialized KPipeline object
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Warning: CUDA not detected, falling back to CPU")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        pipeline = KPipeline(lang_code=lang_code)
        if hasattr(pipeline, 'device'):
            print(f"Pipeline initialized on device: {pipeline.device}")
        return pipeline
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

def check_gpu_status():
    """
    Print GPU information if available.
    """
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"CUDA Available: Yes")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("\nNo GPU detected, running on CPU")

def validate_voice(voice: str, pipeline: KPipeline) -> str:
    """
    Validate if the provided voice is supported by attempting to load it.

    :param voice: The voice identifier
    :param pipeline: An initialized TTS pipeline
    :return: The validated voice string
    """
    try:
        pipeline.load_voice(voice)
        return voice
    except Exception as e:
        print(f"Error validating voice: {e}")
        sys.exit(1)
