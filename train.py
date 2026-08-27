import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_FILE = Path("training_data.txt")
OUTPUT_FILE = Path("www/model.json")

WINDOW_SIZE = 4
MAX_WORDS = 30000


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9']+|[.,!?;:]", text)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("training_data.txt was not found.")

    text = DATA_FILE.read_text(encoding="utf-8")

    tokens = tokenize(text)

    if len(tokens) < 100:
        raise ValueError(
            "training_data.txt contains too little training data. "
            "Add more text."
        )

    print(f"Training tokens: {len(tokens):,}")

    # Keep the most common words.
    counts = Counter(tokens)

    vocabulary = [
        word for word, _ in counts.most_common(MAX_WORDS)
    ]

    vocab_set = set(vocabulary)

    # Word -> words that commonly appear near it.
    associations = defaultdict(Counter)

    for i, word in enumerate(tokens):

        if word not in vocab_set:
            continue

        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)

        for j in range(start, end):

            if i == j:
                continue

            nearby = tokens[j]

            if nearby in vocab_set:
                associations[word][nearby] += 1

    # Convert Counters into compact probability-like scores.
    model = {}

    for word, counter in associations.items():

        total = sum(counter.values())

        if total == 0:
            continue

        related = []

        for next_word, count in counter.most_common(30):

            probability = count / total

            related.append([
                next_word,
                round(probability, 6)
            ])

        model[word] = related

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "name": "MiniAI",
        "version": 1,
        "type": "word_association_language_model",
        "vocabulary_size": len(vocabulary),
        "training_tokens": len(tokens),
        "window_size": WINDOW_SIZE,
        "associations": model
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            separators=(",", ":")
        ),
        encoding="utf-8"
    )

    print()
    print("Training complete.")
    print(f"Vocabulary: {len(vocabulary):,}")
    print(f"Model saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
