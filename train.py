import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MINI AI — TINY GPT + ONNX EXPORT
# ============================================================

DATA_FILE = Path("training_data.txt")
OUTPUT_DIR = Path("www")

MODEL_FILE = OUTPUT_DIR / "model.pt"
ONNX_FILE = OUTPUT_DIR / "model.onnx"
VOCAB_FILE = OUTPUT_DIR / "vocab.json"

BLOCK_SIZE = 64

EMBED_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2

DROPOUT = 0.0

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPOCHS = 10

MAX_VOCAB = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    text = text.lower()

    return re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "training_data.txt was not found."
    )

text = DATA_FILE.read_text(
    encoding="utf-8"
)

tokens = tokenize(text)

if len(tokens) <= BLOCK_SIZE + 1:
    raise ValueError(
        "training_data.txt is too small. "
        f"Add more than {BLOCK_SIZE + 1} tokens."
    )

print("=" * 60)
print("MINIAI NEURAL TRAINER")
print("=" * 60)

print("Device:", DEVICE)
print("Training tokens:", len(tokens))


# ============================================================
# VOCABULARY
# ============================================================

counts = {}

for token in tokens:
    counts[token] = counts.get(token, 0) + 1


special_tokens = [
    "<PAD>",
    "<UNK>"
]


sorted_tokens = sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True
)


vocabulary = special_tokens + [
    token
    for token, _ in sorted_tokens
    if token not in special_tokens
][:MAX_VOCAB - len(special_tokens)]


stoi = {
    token: i
    for i, token in enumerate(vocabulary)
}


itos = {
    i: token
    for token, i in stoi.items()
}


encoded = [
    stoi.get(
        token,
        stoi["<UNK>"]
    )
    for token in tokens
]


data = torch.tensor(
    encoded,
    dtype=torch.long
)


print("Vocabulary:", len(vocabulary))


# ============================================================
# DATA BATCH
# ============================================================

def get_batch():

    max_start = len(data) - BLOCK_SIZE - 1

    starts = torch.randint(
        0,
        max_start + 1,
        (BATCH_SIZE,)
    )

    x = torch.stack(
        [
            data[
                i:i + BLOCK_SIZE
            ]
            for i in starts
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1:i + BLOCK_SIZE + 1
            ]
            for i in starts
        ]
    )

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


# ============================================================
# ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embed_size,
        num_heads,
        block_size
    ):

        super().__init__()

        if embed_size % num_heads != 0:
            raise ValueError(
                "EMBED_SIZE must be divisible by NUM_HEADS."
            )

        self.num_heads = num_heads

        self.head_size = (
            embed_size // num_heads
        )

        self.qkv = nn.Linear(
            embed_size,
            embed_size * 3
        )

        self.projection = nn.Linear(
            embed_size,
            embed_size
        )

        mask = torch.tril(
            torch.ones(
                block_size,
                block_size
            )
        )

        self.register_buffer(
            "mask",
            mask
        )


    def forward(self, x):

        B, T, C = x.shape

        q, k, v = self.qkv(
            x
        ).chunk(
            3,
            dim=-1
        )

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        attention = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_size
        )

        attention = attention.masked_fill(
            self.mask[
                :T,
                :T
            ] == 0,
            float("-inf")
        )

        attention = F.softmax(
            attention,
            dim=-1
        )

        output = attention @ v

        output = output.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        return self.projection(
            output
        )


# ============================================================
# FEED FORWARD
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, embed_size):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                embed_size,
                embed_size * 4
            ),

            nn.GELU(),

            nn.Linear(
                embed_size * 4,
                embed_size
            )
        )


    def forward(self, x):

        return self.net(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_size,
        num_heads,
        block_size
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_size
        )

        self.attention = CausalSelfAttention(
            embed_size,
            num_heads,
            block_size
        )

        self.norm2 = nn.LayerNorm(
            embed_size
        )

        self.feedforward = FeedForward(
            embed_size
        )


    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feedforward(
            self.norm2(x)
        )

        return x


# ============================================================
# MINI GPT
# ============================================================

class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        num_layers,
        block_size
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embed_size
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    block_size
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(
            embed_size
        )

        self.output = nn.Linear(
            embed_size,
            vocab_size,
            bias=False
        )

        self.block_size = block_size


    def forward(self, index):

        B, T = index.shape

        positions = torch.arange(
            T,
            device=index.device
        )

        x = (
            self.token_embedding(index)
            +
            self.position_embedding(
                positions
            )
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.output(x)

        return logits


# ============================================================
# CREATE MODEL
# ============================================================

model = MiniGPT(
    vocab_size=len(vocabulary),
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE
).to(DEVICE)


parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Parameters:",
    f"{parameters:,}"
)


# ============================================================
# TRAIN
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


model.train()


steps_per_epoch = max(
    1,
    len(data) //
    (BLOCK_SIZE * BATCH_SIZE)
)


print()
print("Starting training...")


for epoch in range(EPOCHS):

    total_loss = 0.0

    for step in range(
        steps_per_epoch
    ):

        x, y = get_batch()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()


    average_loss = (
        total_loss /
        steps_per_epoch
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# SAVE PYTORCH MODEL
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


torch.save(
    {
        "model_state": model.state_dict(),

        "config": {
            "vocab_size": len(vocabulary),
            "embed_size": EMBED_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "block_size": BLOCK_SIZE
        }
    },
    MODEL_FILE
)
# ============================================================
# SAVE MODEL CONFIGURATION
# ============================================================

CONFIG_FILE = OUTPUT_DIR / "model_config.json"

CONFIG_FILE.write_text(
    json.dumps(
        {
            "block_size": BLOCK_SIZE,
            "embed_size": EMBED_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "vocab_size": len(vocabulary)
        },
        indent=2
    ),
    encoding="utf-8"
)

# ============================================================
# SAVE VOCABULARY
# ============================================================

VOCAB_FILE.write_text(
    json.dumps(
        {
            "stoi": stoi,
            "itos": {
                str(i): token
                for i, token in itos.items()
            }
        },
        ensure_ascii=False
    ),
    encoding="utf-8"
)

# ============================================================
# ONNX EXPORT
# ============================================================

print()
print("Exporting ONNX model...")

model.eval()

# ONNX export uses the model's full context length.
# The Android app will pad shorter prompts to this size.

dummy_input = torch.zeros(
    (1, BLOCK_SIZE),
    dtype=torch.long,
    device=DEVICE
)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_FILE,
    input_names=["input_ids"],
    output_names=["logits"],
    opset_version=17
)

print(
    "ONNX export complete:",
    ONNX_FILE
)
