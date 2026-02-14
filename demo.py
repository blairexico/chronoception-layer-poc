"""Demo script showing both temporal intervention methods.

Usage:
    python demo.py                    # Context injection only (Method A)
    python demo.py --intervention     # Both methods (A + B logits steering)
"""

import argparse
import logging
import sys

from chronoception import ChronoceptionChat, ChronoceptionConfig, InterventionConfig


def run_demo(use_intervention: bool = False) -> None:
    config = ChronoceptionConfig(
        intervention=InterventionConfig(enabled=use_intervention),
    )
    # Use local SQLite for demo
    config.database.db_path = "demo_temporal.db"

    chat = ChronoceptionChat(config)
    chat.initialize()

    user_id = "demo_user"

    print("\n" + "=" * 60)
    print("CHRONOCEPTION - TEMPORAL AWARENESS DEMO")
    print(f"Method A (Context Injection): ON")
    print(f"Method B (Logits Steering):   {'ON' if use_intervention else 'OFF'}")
    print("=" * 60)

    # Turn 1: User mentions starting a blog
    print("\n\n--- TURN 1 ---")
    print("User: I started my blog on January 15, 2024\n")

    result1 = chat.chat(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"}
        ],
    )

    print(f"Assistant: {result1.response}")
    print(f"[Learned {result1.learned_facts} facts]")

    # Turn 2: User mentions another event
    print("\n\n--- TURN 2 ---")
    print("User: I also moved to a new city on June 1st, 2024\n")

    result2 = chat.chat(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": result1.response},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
        ],
    )

    print(f"Assistant: {result2.response}")
    print(f"[Learned {result2.learned_facts} new, {result2.total_facts} total facts]")

    # Turn 3: Query temporal fact
    print("\n\n--- TURN 3 (temporal query) ---")
    print("User: How long has it been since I started my blog?\n")

    result3 = chat.chat(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": result1.response},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
            {"role": "assistant", "content": result2.response},
            {"role": "user", "content": "How long has it been since I started my blog?"},
        ],
    )

    print(f"Assistant: {result3.response}")
    print(f"\n--- Temporal Context ---")
    print(result3.temporal_context)
    print(f"Intervention active: {result3.intervention_active}")

    # Turn 4: Tangential usage (should weave in temporal awareness)
    print("\n\n--- TURN 4 (tangential) ---")
    print("User: Should I start monetizing my blog?\n")

    result4 = chat.chat(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": result1.response},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
            {"role": "assistant", "content": result2.response},
            {"role": "user", "content": "How long has it been since I started my blog?"},
            {"role": "assistant", "content": result3.response},
            {"role": "user", "content": "Should I start monetizing my blog?"},
        ],
    )

    print(f"Assistant: {result4.response}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    chat.close()


def main():
    parser = argparse.ArgumentParser(description="Chronoception demo")
    parser.add_argument(
        "--intervention", action="store_true",
        help="Enable logits steering (Method B) in addition to context injection",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_demo(use_intervention=args.intervention)


if __name__ == "__main__":
    main()
