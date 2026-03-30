# ViLM2CCG Training Dataset

Training dataset for fine-tuning a local language model (e.g. via **LM Studio**) on the
**NL → CCG** (Natural Language → Combinational Circuit Graph) task in Vietnamese.

---

## Dataset overview

| Split | Examples |
|---|---|
| `train` | 182 |
| `val` | 23 |
| `test` | 23 |
| **Total** | **228** |

### Class distribution (full dataset)

| Circuit type | Count |
|---|---|
| `and` | 32 |
| `or` | 24 |
| `mux` | 30 |
| `nand` | 18 |
| `nor` | 18 |
| `xor` | 18 |
| `not` | 10 |
| `xnor` | 8 |
| `buf` | 6 |
| `decoder` | 16 |
| `encoder` | 14 |
| `half_adder` | 12 |
| `full_adder` | 12 |
| `comparator` | 10 |

Splits are **stratified by circuit type** to maintain class balance across train/val/test.

---

## File structure

```
dataset/
├── build_dataset.py          # Generator script (re-run to regenerate)
├── README.md                 # This file
└── data/
    ├── full.jsonl            # All 228 examples (Alpaca format)
    ├── train.jsonl           # Training split (Alpaca format)
    ├── val.jsonl             # Validation split (Alpaca format)
    ├── test.jsonl            # Test split (Alpaca format)
    ├── full_chat.jsonl       # All examples (chat/messages format)
    ├── train_chat.jsonl      # Training split (chat format)
    ├── val_chat.jsonl        # Validation split (chat format)
    └── test_chat.jsonl       # Test split (chat format)
```

---

## Data formats

### Alpaca instruction format (`*.jsonl`)

Each line is a JSON object:

```json
{
  "instruction": "Bạn là một chuyên gia thiết kế mạch số. ...",
  "input":       "Tạo mạch AND 2 đầu vào",
  "output":      "{\n  \"circuit_name\": \"and_2in\", ... }"
}
```

Use `train.jsonl` / `val.jsonl` / `test.jsonl` for Alpaca-style fine-tuning
(e.g. with [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)).

### Chat / messages format (`*_chat.jsonl`)

Each line is a JSON object with a `messages` array, compatible with
OpenAI fine-tuning and **LM Studio**:

```json
{
  "messages": [
    {"role": "system",    "content": "Bạn là một chuyên gia thiết kế mạch số. ..."},
    {"role": "user",      "content": "Tạo mạch AND 2 đầu vào"},
    {"role": "assistant", "content": "{\n  \"circuit_name\": \"and_2in\", ... }"}
  ]
}
```

Use `train_chat.jsonl` for fine-tuning in LM Studio or with the OpenAI API.

---

## Regenerating the dataset

```bash
# From the repository root
python dataset/build_dataset.py

# Custom output directory
python dataset/build_dataset.py --out /path/to/output

# Adjust split ratios or random seed
python dataset/build_dataset.py --val-ratio 0.1 --test-ratio 0.1 --seed 42
```

---

## Fine-tuning with LM Studio

1. Open **LM Studio** → **My Models** → select your base model.
2. Go to the **Fine-tuning** (or **Training**) tab.
3. Upload `dataset/data/train_chat.jsonl` as the training file and
   `dataset/data/val_chat.jsonl` as the validation file.
4. Set the system prompt to match the `system` role content in the JSONL.
5. Start training with LoRA or full fine-tuning as supported by your model.

After fine-tuning, load the new adapter in LM Studio and enable
**Sử dụng LM Studio** in the ViLM2CCG Streamlit sidebar.

---

## Example prompts covered

| Vietnamese | English alias | Circuit type |
|---|---|---|
| Tạo mạch AND 2 đầu vào | AND gate 2 input | `and` |
| Tạo mạch MUX 4-1 | MUX 4-to-1 | `mux` |
| Tạo bộ cộng nửa | half adder | `half_adder` |
| Tạo bộ cộng đầy đủ | full adder | `full_adder` |
| Bộ giải mã 2 vào 4 ra | decoder 2-to-4 | `decoder` |
| Bộ mã hóa 4 vào 2 ra | encoder 4-to-2 | `encoder` |
| Bộ so sánh 1 bit | 1-bit comparator | `comparator` |
| Cổng NAND 3 đầu vào | NAND gate 3 input | `nand` |
| Cổng XOR 2 đầu vào | XOR gate 2 input | `xor` |
| Cổng NOT | NOT gate / inverter | `not` |

Each circuit type has 6–10 Vietnamese phrasing variations to improve
generalisation across different input styles.
