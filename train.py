import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MINI AI — TINY GPT-STYLE LANGUAGE MODEL
# ============================================================

DATA_FILE = Path("training_data.txt")
OUTPUT_DIR = Path("www")
MODEL_FILE = OUTPUT_DIR / "model.pt"
VOCAB_FILE = OUTPUT_DIR / "vocab.json"

# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

BLOCK_SIZE = 128

EMBED_SIZE = 128
NUM_HEADS = 4
NUM_LAYERS = 4

DROPOUT = 0.1

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 10

# Maximum vocabulary size
MAX_VOCAB = 20000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Simple tokenizer.

    It separates words and punctuation.

    Example:

        Hello, world!

    becomes:

        ["hello", ",", "world", "!"]
    """

    text = text.lower()

    return re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        "training_data.txt was not found."
    )


text = DATA_FILE.read_text(
    encoding="utf-8"
)

if len(text.strip()) < 500:

    raise ValueError(
        "training_data.txt contains too little text."
    )


tokens = tokenize(text)

print("=" * 60)
print("MINI AI — NEURAL TRAINER")
print("=" * 60)

print("Device:", DEVICE)
print("Characters:", len(text))
print("Tokens:", len(tokens))


# ============================================================
# BUILD VOCABULARY
# ============================================================

special_tokens = [
    "<PAD>",
    "<UNK>"
]

counts = {}

for token in tokens:

    counts[token] = counts.get(
        token,
        0
    ) + 1


sorted_words = sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True
)


vocabulary = special_tokens + [
    word
    for word, count in sorted_words
    if word not in special_tokens
][:MAX_VOCAB - len(special_tokens)]


stoi = {
    token: index
    for index, token in enumerate(vocabulary)
}


itos = {
    index: token
    for token, index in stoi.items()
}


encoded = [
    stoi.get(
        token,
        stoi["<UNK>"]
    )
    for token in tokens
]


print("Vocabulary:", len(vocabulary))


# ============================================================
# DATASET
# ============================================================

data = torch.tensor(
    encoded,
    dtype=torch.long
)


if len(data) <= BLOCK_SIZE + 1:

    raise ValueError(
        f"Training data must contain more than "
        f"{BLOCK_SIZE + 1} tokens."
    )


def get_batch():

    max_start = len(data) - BLOCK_SIZE - 1

    starts = torch.randint(
        0,
        max_start,
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
# CAUSAL SELF-ATTENTION
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


        self.dropout = nn.Dropout(
            DROPOUT
        )


        mask = torch.tril(
            torch.ones(
                block_size,
                block_size
            )
        )


        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                block_size,
                block_size
            )
        )


    def forward(self, x):

        batch_size, sequence_length, channels = x.shape


        qkv = self.qkv(x)


        q, k, v = qkv.chunk(
            3,
            dim=-1
        )


        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)


        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)


        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)


        attention = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_size)


        attention = attention.masked_fill(
            self.mask[
                :,
                :,
                :sequence_length,
                :sequence_length
            ] == 0,
            float("-inf")
        )


        attention = F.softmax(
            attention,
            dim=-1
        )


        attention = self.dropout(
            attention
        )


        output = attention @ v


        output = output.transpose(
            1,
            2
        ).contiguous().view(
            batch_size,
            sequence_length,
            channels
        )


        return self.projection(
            output
        )


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, embed_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embed_size,
                embed_size * 4
            ),

            nn.GELU(),

            nn.Linear(
                embed_size * 4,
                embed_size
            ),

            nn.Dropout(
                DROPOUT
            )
        )


    def forward(self, x):

        return self.network(x)


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

        self.feed_forward = FeedForward(
            embed_size
        )


    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


# ============================================================
# TINY GPT
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


        self.final_norm = nn.LayerNorm(
            embed_size
        )


        self.output = nn.Linear(
            embed_size,
            vocab_size,
            bias=False
        )


        # Weight tying:
        # input and output embeddings share weights.

        self.output.weight = (
            self.token_embedding.weight
        )


        self.block_size = block_size


        self.apply(
            self._initialize_weights
        )


    def _initialize_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )


        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


    def forward(
        self,
        index,
        targets=None
    ):

        batch_size, sequence_length = (
            index.shape
        )


        if sequence_length > self.block_size:

            raise ValueError(
                "Sequence is longer than BLOCK_SIZE."
            )


        positions = torch.arange(
            sequence_length,
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


        x = self.final_norm(x)


        logits = self.output(x)


        loss = None


        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),

                targets.reshape(
                    -1
                )
            )


        return logits, loss


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


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print(
    "Parameters:",
    f"{parameter_count:,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)


model.train()


steps_per_epoch = max(
    1,
    len(data) // (
        BLOCK_SIZE * BATCH_SIZE
    )
)


for epoch in range(EPOCHS):

    total_loss = 0.0


    for step in range(
        steps_per_epoch
    ):

        x, y = get_batch()


        optimizer.zero_grad(
            set_to_none=True
        )


        logits, loss = model(
            x,
            y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )


        optimizer.step()


        total_loss += loss.item()


        if (
            step == 0
            or
            (step + 1)
            % max(
                1,
                steps_per_epoch // 5
            )
            == 0
        ):

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"| Step {step + 1}/{steps_per_epoch} "
                f"| Loss {loss.item():.4f}"
            )


    average_loss = (
        total_loss /
        steps_per_epoch
    )


    print(
        f"Epoch {epoch + 1} complete "
        f"| Average loss: "
        f"{average_loss:.4f}"
    )


# ============================================================
# SAVE MODEL
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
# SAVE VOCABULARY
# ============================================================

VOCAB_FILE.write_text(
    json.dumps(
        {
            "stoi": stoi,
            "itos": {
                str(index): token
                for index, token in itos.items()
            }
        },
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Model:",
    MODEL_FILE
)

print(
    "Vocabulary:",
    VOCAB_FILE
)

print(
    "Parameters:",
    f"{parameter_count:,}"
)

print()
print(
    "The neural model has been trained."
)
