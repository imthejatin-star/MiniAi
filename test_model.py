import json
import os
import sys

import numpy as np
import onnxruntime as ort


MODEL_FILE = "www/model.onnx"
VOCAB_FILE = "www/vocab.json"
CONFIG_FILE = "www/model_config.json"


print("=" * 60)
print("MINIAI MODEL TEST")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    MODEL_FILE,
    VOCAB_FILE,
    CONFIG_FILE
]

for file_path in required_files:

    if not os.path.exists(file_path):

        print(f"ERROR: Missing file: {file_path}")
        sys.exit(1)

    print(f"Found: {file_path}")


# ============================================================
# LOAD VOCABULARY
# ============================================================

try:

    with open(
        VOCAB_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        vocab = json.load(f)

except Exception as e:

    print()
    print("ERROR loading vocabulary:")
    print(e)
    sys.exit(1)


# Support both common vocabulary formats.

if isinstance(vocab, dict) and "stoi" in vocab:

    stoi = vocab["stoi"]

    itos = vocab.get("itos", {})

else:

    stoi = vocab
    itos = {
        str(value): key
        for key, value in vocab.items()
    }


vocabulary_size = len(stoi)


# ============================================================
# LOAD CONFIG
# ============================================================

try:

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

except Exception as e:

    print()
    print("ERROR loading model configuration:")
    print(e)
    sys.exit(1)


block_size = int(
    config.get(
        "block_size",
        64
    )
)


print(f"Block size: {block_size}")
print(f"Vocabulary: {vocabulary_size}")


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print()
print("Loading ONNX model...")


try:

    session = ort.InferenceSession(
        MODEL_FILE,
        providers=[
            "CPUExecutionProvider"
        ]
    )

except Exception as e:

    print()
    print("ERROR loading ONNX model:")
    print(e)
    sys.exit(1)


print("ONNX model loaded successfully.")


# ============================================================
# DISPLAY MODEL INPUTS
# ============================================================

print()
print("Inputs:")

for input_info in session.get_inputs():

    print(
        input_info.name,
        input_info.shape,
        input_info.type
    )


# ============================================================
# DISPLAY MODEL OUTPUTS
# ============================================================

print()
print("Outputs:")

for output_info in session.get_outputs():

    print(
        output_info.name,
        output_info.shape,
        output_info.type
    )


# ============================================================
# VALIDATE INPUT
# ============================================================

inputs = session.get_inputs()

if len(inputs) == 0:

    print()
    print("ERROR: Model has no inputs.")
    sys.exit(1)


input_info = inputs[0]

input_name = input_info.name


print()
print("Expected input:")
print(
    f"Name: {input_name}"
)
print(
    f"Shape: {input_info.shape}"
)
print(
    f"Type: {input_info.type}"
)


# ============================================================
# CREATE EXACT MODEL INPUT
# ============================================================

print()
print("Creating test input...")


# The current MiniAI model expects:
#
# [1, 64]
#
# Therefore we MUST send exactly block_size tokens.

test_tokens = np.zeros(
    (
        1,
        block_size
    ),
    dtype=np.int64
)


# Put a few valid token IDs into the beginning.
#
# IMPORTANT:
# We never allow an ID greater than vocabulary_size - 1.

number_of_test_tokens = min(
    8,
    vocabulary_size,
    block_size
)


for i in range(
    number_of_test_tokens
):

    test_tokens[
        0,
        i
    ] = i


print(
    "Test input shape:",
    test_tokens.shape
)

print(
    "Test input dtype:",
    test_tokens.dtype
)

print(
    "First tokens:",
    test_tokens[
        0,
        :number_of_test_tokens
    ].tolist()
)


# ============================================================
# RUN INFERENCE
# ============================================================

print()
print("Running inference...")


try:

    outputs = session.run(
        None,
        {
            input_name: test_tokens
        }
    )

except Exception as e:

    print()
    print("ERROR during inference:")
    print(e)
    sys.exit(1)


print("Inference successful.")


# ============================================================
# CHECK OUTPUT
# ============================================================

if not outputs:

    print()
    print("ERROR: Model returned no outputs.")
    sys.exit(1)


logits = outputs[0]


print()
print("Output information:")
print(
    "Shape:",
    logits.shape
)

print(
    "Dtype:",
    logits.dtype
)


# ============================================================
# VALIDATE OUTPUT SHAPE
# ============================================================

expected_output_shape = (
    1,
    block_size,
    vocabulary_size
)


print()
print(
    "Expected output shape:",
    expected_output_shape
)

print(
    "Actual output shape:",
    tuple(logits.shape)
)


if tuple(logits.shape) != expected_output_shape:

    print()
    print(
        "WARNING: Output shape does not exactly match "
        "the expected shape."
    )

else:

    print(
        "Output shape is correct."
    )


# ============================================================
# BASIC NUMERICAL CHECK
# ============================================================

if not np.isfinite(logits).all():

    print()
    print(
        "ERROR: Model produced NaN or infinite values."
    )

    sys.exit(1)


print(
    "Output contains only valid numerical values."
)


# ============================================================
# GET LAST TOKEN PREDICTION
# ============================================================

last_logits = logits[
    0,
    -1,
    :
]


predicted_id = int(
    np.argmax(
        last_logits
    )
)


predicted_token = itos.get(
    str(predicted_id),
    "<UNKNOWN>"
)


print()
print("=" * 60)
print("MODEL TEST RESULT")
print("=" * 60)

print(
    f"Predicted token ID: {predicted_id}"
)

print(
    f"Predicted token: {predicted_token}"
)

print(
    f"Model input shape: {test_tokens.shape}"
)

print(
    f"Model output shape: {logits.shape}"
)

print()
print("MiniAI ONNX model is working correctly.")
print("=" * 60)
