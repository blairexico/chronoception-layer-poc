"""Local demo: run without Modal.

Usage:
    python demo.py                    # Context injection only (Method A)
    python demo.py --intervention     # Both methods (A + B logits steering)
"""

import argparse
import logging

from chronoception import ChronoceptionChat


def run_demo(use_intervention: bool = False) -> None:
    chat = ChronoceptionChat(db_path="demo_temporal.db")
    chat.initialize()

    user_id = "demo_user"

    print("\n" + "=" * 60)
    print("CHRONOCEPTION DEMO")
    print(f"Method A (Context Injection): ON")
    print(f"Method B (Logits Steering):   {'ON' if use_intervention else 'OFF'}")
    print("=" * 60)

    # Turn 1: User mentions starting a blog
    print("\n--- TURN 1 ---")
    print("User: I started my blog on January 15, 2024\n")
    r1 = chat.chat(
        user_id, [{"role": "user", "content": "I started my blog on January 15, 2024"}],
        use_intervention=use_intervention,
    )
    print(f"Assistant: {r1.response}")
    print(f"[Learned {r1.learned_facts} facts]")

    # Turn 2: Another event
    print("\n--- TURN 2 ---")
    print("User: I also moved to a new city on June 1st, 2024\n")
    r2 = chat.chat(
        user_id,
        [
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": r1.response},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
        ],
        use_intervention=use_intervention,
    )
    print(f"Assistant: {r2.response}")
    print(f"[Learned {r2.learned_facts} new, {r2.total_facts} total]")

    # Turn 3: Temporal query
    print("\n--- TURN 3 (temporal query) ---")
    print("User: How long has it been since I started my blog?\n")
    r3 = chat.chat(
        user_id,
        [
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": r1.response},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
            {"role": "assistant", "content": r2.response},
            {"role": "user", "content": "How long has it been since I started my blog?"},
        ],
        use_intervention=use_intervention,
    )
    print(f"Assistant: {r3.response}")
    print(f"\n--- Temporal Context ---\n{r3.temporal_context}")

    # Turn 4: Tangential
    print("\n--- TURN 4 (tangential) ---")
    print("User: Should I start monetizing my blog?\n")
    r4 = chat.chat(
        user_id,
        [
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": r1.response},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
            {"role": "assistant", "content": r2.response},
            {"role": "user", "content": "How long has it been since I started my blog?"},
            {"role": "assistant", "content": r3.response},
            {"role": "user", "content": "Should I start monetizing my blog?"},
        ],
        use_intervention=use_intervention,
    )
    print(f"Assistant: {r4.response}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    chat.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronoception demo")
    parser.add_argument("--intervention", action="store_true", help="Enable logits steering")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_demo(use_intervention=args.intervention)
