"""Modal deployment for the Chronoception temporal awareness system.

Deploys the chat function as a Modal app with GPU support and
persistent fact storage via Modal volumes.

Usage:
    modal run modal_app.py          # Run the demo
    modal deploy modal_app.py       # Deploy as a persistent app
"""

import modal
from modal import Volume

app = modal.App("chronoception")

volume = Volume.from_name("temporal-facts-db", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "accelerate",
    "bitsandbytes",
)


@app.function(
    gpu="T4",
    volumes={"/data": volume},
    image=image,
    timeout=900,
)
def chat(user_id: str, messages: list, use_intervention: bool = False):
    """Multi-turn conversation with temporal awareness.

    Args:
        user_id: Unique user identifier.
        messages: List of {"role": "user"/"assistant", "content": "..."}.
        use_intervention: Enable logits steering (Method B).

    Returns:
        Dict with response, learned_facts, total_facts, temporal_context.
    """
    import sys
    sys.path.insert(0, "/root")

    from chronoception import ChronoceptionChat, ChronoceptionConfig, InterventionConfig

    config = ChronoceptionConfig(
        intervention=InterventionConfig(enabled=use_intervention),
    )

    chat_engine = ChronoceptionChat(config)
    chat_engine.initialize()

    result = chat_engine.chat(user_id=user_id, messages=messages)

    chat_engine.close()
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
    """Demo multi-turn conversation with temporal learning."""

    user_id = "demo_user"

    print("\n" + "=" * 60)
    print("CHRONOCEPTION - MODAL DEPLOYMENT DEMO")
    print("=" * 60)

    # Turn 1
    print("\n--- TURN 1 ---")
    result1 = chat.remote(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"}
        ],
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
