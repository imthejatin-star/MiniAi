import json
import re
from pathlib import Path


INPUT_FILE = Path("study_input.txt")
OUTPUT_FILE = Path("www/study_data.json")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def make_questions(sentences):
    questions = []

    for sentence in sentences:
        words = sentence.split()

        if len(words) < 7:
            continue

        # Definition-style sentence
        match = re.match(
            r"(.+?)\s+(?:is|are|means|refers to)\s+(.+)",
            sentence,
            re.IGNORECASE
        )

        if match:
            subject = clean_text(match.group(1))
            explanation = clean_text(match.group(2))

            questions.append({
                "question": f"What is {subject}?",
                "answer": sentence
            })

        # Requirement / component sentence
        elif "requirements" in sentence.lower():
            questions.append({
                "question": "What are the main requirements mentioned?",
                "answer": sentence
            })

        # Process sentence
        elif any(
            word in sentence.lower()
            for word in [
                "process",
                "during",
                "produced",
                "released",
                "absorbs"
            ]
        ):
            questions.append({
                "question": "What does this statement explain?",
                "answer": sentence
            })

    return questions[:20]


def make_flashcards(sentences):
    flashcards = []

    for sentence in sentences:
        match = re.match(
            r"(.+?)\s+(?:is|are|means|refers to)\s+(.+)",
            sentence,
            re.IGNORECASE
        )

        if match:
            front = clean_text(match.group(1))
            back = clean_text(match.group(2))

            flashcards.append({
                "front": front,
                "back": back
            })

    return flashcards[:20]


def make_key_points(sentences):
    points = []

    for sentence in sentences:
        sentence = clean_text(sentence)

        if len(sentence) >= 30:
            points.append(sentence)

    return points[:20]


def make_quiz(sentences):
    quiz = []

    for index, sentence in enumerate(sentences[:10]):

        words = sentence.split()

        if len(words) < 8:
            continue

        answer_word = words[-1].rstrip(".,!?")

        if len(answer_word) < 4:
            continue

        masked = sentence.replace(
            answer_word,
            "_____",
            1
        )

        options = [
            answer_word
        ]

        for other_sentence in sentences:
            other_words = other_sentence.split()

            for word in other_words:
                candidate = word.rstrip(".,!?")

                if (
                    len(candidate) >= 4
                    and candidate.lower() != answer_word.lower()
                    and candidate not in options
                ):
                    options.append(candidate)

                if len(options) >= 4:
                    break

            if len(options) >= 4:
                break

        while len(options) < 4:
            options.append("None of these")

        quiz.append({
            "question": masked,
            "options": options[:4],
            "answer": answer_word
        })

    return quiz


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "study_input.txt was not found."
        )

    raw_text = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    if not raw_text.strip():
        raise ValueError(
            "study_input.txt is empty."
        )

    title = "Study Topic"

    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    if lines:
        title = lines[0]

    body = "\n".join(lines[1:])

    sentences = split_sentences(body)

    data = {
        "title": title,
        "summary": clean_text(body),
        "key_points": make_key_points(sentences),
        "flashcards": make_flashcards(sentences),
        "questions": make_questions(sentences),
        "quiz": make_quiz(sentences)
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("Study material processed successfully.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Key points: {len(data['key_points'])}")
    print(f"Flashcards: {len(data['flashcards'])}")
    print(f"Questions: {len(data['questions'])}")
    print(f"Quiz questions: {len(data['quiz'])}")


if __name__ == "__main__":
    main()
