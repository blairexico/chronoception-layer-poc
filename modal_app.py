"""Modal deployment for Chronoception.

Usage:
    modal run modal_app.py
    modal deploy modal_app.py
"""

import modal
from modal import Volume

app = modal.App("chronoception")

volume = Volume.from_name("temporal-facts-db", create_if_missing=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def download_model():
    """Pre-download model weights at image build time."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Downloading {MODEL_NAME}...")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Model cached.")


image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "bitsandbytes",
    )
    .run_function(download_model)
    .add_local_file("chronoception.py", "/root/chronoception.py")
    .add_local_file("intervention.py", "/root/intervention.py")
)


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    image=image,
    timeout=900,
)
def chat(user_id: str, messages: list, use_intervention: bool = False):
    """Multi-turn conversation with temporal awareness."""
    from chronoception import ChronoceptionChat

    print(f"[chronoception] Loading model: {MODEL_NAME}")
    engine = ChronoceptionChat(model_name=MODEL_NAME)
    engine.initialize()
    print("[chronoception] Model loaded, processing turn...")

    result = engine.chat(user_id=user_id, messages=messages, use_intervention=use_intervention)
    print(f"[chronoception] Done. Learned {result.learned_facts} facts, {result.total_facts} total.")

    engine.close()
    volume.commit()

    return {
        "response": result.response,
        "learned_facts": result.learned_facts,
        "total_facts": result.total_facts,
        "temporal_context": result.temporal_context,
        "intervention_active": result.intervention_active,
    }


@app.local_entrypoint()
def demo():
    user_id = "demo_user"

    print("\n" + "=" * 60)
    print("CHRONOCEPTION - MODAL DEMO")
    print("=" * 60)

    # Turn 1
    print("\n--- TURN 1 ---")
    result1 = chat.remote(
        user_id=user_id,
        messages=[{"role": "user", "content": "I started my blog on January 15, 2024"}],
    )
    print(f"Assistant: {result1['response']}")
    print(f"Learned: {result1['learned_facts']} facts")

    # Turn 2
    print("\n--- TURN 2 ---")
    result2 = chat.remote(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": result1['response']},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
        ],
    )
    print(f"Assistant: {result2['response']}")
    print(f"Learned: {result2['learned_facts']} new facts")

    # Turn 3
    print("\n--- TURN 3 ---")
    result3 = chat.remote(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": result1['response']},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
            {"role": "assistant", "content": result2['response']},
            {"role": "user", "content": "How long has it been since I started my blog?"},
        ],
    )
    print(f"Assistant: {result3['response']}")
    print(f"\nTemporal Context:\n{result3['temporal_context']}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
