import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_FILE = Path("www/model.onnx")
VOCAB_FILE = Path("www/vocab.json")
CONFIG_FILE = Path("www/model_config.json")


print("=" * 60)
print("MINIAI MODEL TEST")
print("=" * 60)


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

for file in [
    MODEL_FILE,
    VOCAB_FILE,
    CONFIG_FILE
]:

    if not file.exists():

        raise FileNotFoundError(
            f"Missing required file: {file}"
        )

    print(
        "Found:",
        file
    )


# ------------------------------------------------------------
# LOAD CONFIG
# ------------------------------------------------------------

config = json.loads(
    CONFIG_FILE.read_text(
        encoding="utf-8"
    )
)


block_size = int(
    config["block_size"]
)

vocab_size = int(
    config["vocab_size"]
)


print(
    "Block size:",
    block_size
)

print(
    "Vocabulary:",
    vocab_size
)


# ------------------------------------------------------------
# LOAD VOCAB
# ------------------------------------------------------------

vocab = json.loads(
    VOCAB_FILE.read_text(
        encoding="utf-8"
    )
)


if "stoi" not in vocab:
    raise ValueError(
        "vocab.json does not contain stoi."
    )

if "itos" not in vocab:
    raise ValueError(
        "vocab.json does not contain itos."
    )


# ------------------------------------------------------------
# LOAD ONNX
# ------------------------------------------------------------

print()
print("Loading ONNX model...")


session = ort.InferenceSession(
    str(MODEL_FILE),
    providers=[
        "CPUExecutionProvider"
    ]
)


print(
    "ONNX model loaded successfully."
)


# ------------------------------------------------------------
# DISPLAY INPUTS / OUTPUTS
# ------------------------------------------------------------

print()
print("Inputs:")

for input_info in session.get_inputs():

    print(
        input_info.name,
        input_info.shape,
        input_info.type
    )


print()
print("Outputs:")

for output_info in session.get_outputs():

    print(
        output_info.name,
        output_info.shape,
        output_info.type
    )


# ------------------------------------------------------------
# TEST INFERENCE
# ------------------------------------------------------------

test_length = min(
    8,
    block_size
)


input_ids = np.zeros(
    (
        1,
        test_length
    ),
    dtype=np.int64
)


print()
print(
    "Running inference..."
)


outputs = session.run(
    None,
    {
        "input_ids": input_ids
    }
)


if len(outputs) == 0:

    raise RuntimeError(
        "ONNX model returned no outputs."
    )


logits = outputs[0]


print(
    "Output shape:",
    logits.shape
)


# ------------------------------------------------------------
# VALIDATE OUTPUT
# ------------------------------------------------------------

if logits.ndim != 3:

    raise RuntimeError(
        "Expected logits to have 3 dimensions."
    )


if logits.shape[0] != 1:

    raise RuntimeError(
        "Unexpected batch dimension."
    )


if logits.shape[1] != test_length:

    raise RuntimeError(
        "Unexpected sequence dimension."
    )


if logits.shape[2] != vocab_size:

    raise RuntimeError(
        "Output vocabulary size does not match vocab.json."
    )


if not np.isfinite(logits).all():

    raise RuntimeError(
        "Model produced NaN or infinite values."
    )


print()
print("=" * 60)
print("MODEL TEST PASSED")
print("=" * 60)
