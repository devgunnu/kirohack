"""Model module: Runs inference using TinyLlama via ctransformers."""

import os
from ctransformers import AutoModelForCausalLM

# Path to the GGUF model file, relative to this module
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Lazy-loaded singleton so the model loads once per process
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        print("[Model] Loading TinyLlama... (first call only)")
        _llm = AutoModelForCausalLM.from_pretrained(
            _MODEL_PATH,
            model_type="llama",
            local_files_only=True,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    return _llm


def run_model(prompt: str, simulate_delay: bool = False) -> str:
    """Run inference on the given prompt using TinyLlama.

    Args:
        prompt: A plaintext prompt string.
        simulate_delay: Unused, kept for interface compatibility.

    Returns:
        The model's response as a string.
    """
    llm = _get_llm()

    # TinyLlama chat format
    formatted = (
        "<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{prompt}</s>\n"
        "<|assistant|>\n"
    )

    response = llm(formatted)
    return response.strip()
