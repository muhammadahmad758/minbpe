"""
Roman Urdu vs English: how does a BPE tokenizer treat each language?

Added by Muhammad Ahmad Farooq in this fork.

The two texts below say the SAME thing — one in Roman Urdu (Urdu written
in Latin letters, how most people in Pakistan actually type), one in
English. We train minbpe's BasicTokenizer and compare:

1. Which pieces ("tokens") the tokenizer learns from Roman Urdu text —
   common chunks like "hai", "ka" and "aur" become single tokens.
2. How many tokens each version of the text needs. The counts depend a
   lot on what text the tokenizer was trained on — a tokenizer trained
   on one language's text represents that text more efficiently. That
   is one reason LLMs can behave differently across languages.

Needs the repo's requirements installed first (minbpe imports them):
    pip install -r requirements.txt

Then run it from the repo root:
    python examples/roman_urdu_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minbpe import BasicTokenizer

ROMAN_URDU = (
    "Mera naam Ahmad hai. Main Islamabad mein rehta hoon aur Artificial "
    "Intelligence ki parhai kar raha hoon. Mujhe machine learning bohat "
    "pasand hai kyun ke yeh computers ko data se seekhne ki taqat deti "
    "hai. Har roz main Python practice karta hoon aur chhote chhote "
    "projects banata hoon. Mera khwab hai ke main ek din ek bara AI "
    "engineer banun aur apne mulk ka naam roshan karun. Mehnat aur lagan "
    "se har mushkil asan ho jati hai."
)

ENGLISH = (
    "My name is Ahmad. I live in Islamabad and I am studying Artificial "
    "Intelligence. I really like machine learning because it gives "
    "computers the power to learn from data. Every day I practice Python "
    "and build small projects. My dream is that one day I become a great "
    "AI engineer and make my country proud. With hard work and dedication "
    "every difficulty becomes easy."
)


def show_learned_tokens():
    """Train on Roman Urdu only and show some tokens the tokenizer learns."""
    tok = BasicTokenizer()
    tok.train(ROMAN_URDU, vocab_size=300)
    learned = [tok.vocab[idx] for idx in range(256, 300)]
    readable = [t.decode("utf-8", errors="replace") for t in learned]
    print("Tokens learned from the Roman Urdu text (first 20 of 44):")
    print("  " + ", ".join(repr(t) for t in readable[:20]))
    print()


def compare_token_counts():
    """Train one tokenizer on both texts, then count tokens for each."""
    tok = BasicTokenizer()
    tok.train(ROMAN_URDU + " " + ENGLISH, vocab_size=320)

    print(f"{'':14}{'chars':>7}{'words':>7}{'tokens':>8}{'tokens/word':>13}")
    for name, text in [("Roman Urdu", ROMAN_URDU), ("English", ENGLISH)]:
        ids = tok.encode(text)
        words = len(text.split())
        print(
            f"{name:14}{len(text):>7}{words:>7}{len(ids):>8}"
            f"{len(ids) / words:>13.2f}"
        )

        # round-trip check: encoding then decoding must give the text back
        assert tok.decode(ids) == text, f"round-trip failed for {name}"
    print("\nRound-trip check passed for both texts: decode(encode(x)) == x")


if __name__ == "__main__":
    show_learned_tokens()
    compare_token_counts()
