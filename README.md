# ViLM2CCG

**Vietnamese Language Model to Combinational Circuit Graph**

ViLM2CCG is a Python framework that converts Vietnamese natural-language descriptions of combinational logic circuits into:

- **CircuitJSON** – a structured intermediate representation
- **CCG visualizations** – NetworkX + Matplotlib circuit-graph images (PNG/SVG)
- **Synthesisable Verilog HDL**
- **Functional verification** via Icarus Verilog or a pure-Python simulation fallback
- **Streamlit web dashboard** for interactive use

---

## Directory structure

```
vilm2ccg/
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
├── vilm2ccg/                 # Core package
│   ├── __init__.py
│   ├── circuit_json.py       # CircuitJSON data model
│   ├── input_module.py       # Vietnamese text parser
│   ├── llm_interface.py      # LLM interface + rule-based fallback
│   ├── visualizer.py         # Graph visualizer (NetworkX + Matplotlib)
│   ├── hdl_generator.py      # Verilog transpiler
│   └── verification.py       # Simulation & verification
└── tests/
    └── test_pipeline.py      # Pytest unit tests
```

---

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit demo

```bash
streamlit run app.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## Supported circuits

| Vietnamese prompt | Circuit type |
|---|---|
| Mạch AND N đầu vào | N-input AND gate |
| Cổng OR / NOR / NAND / XOR / XNOR | Basic gates |
| Cổng NOT / bộ đảo | Inverter |
| Mạch MUX 4-1 | 4-to-1 Multiplexer |
| Bộ cộng nửa | Half Adder |
| Bộ cộng đầy đủ | Full Adder |
| Bộ giải mã N đầu vào | N-to-2ᴺ Decoder |
| Bộ mã hóa | Priority Encoder |
| Bộ so sánh | 1-bit Magnitude Comparator |

---

## CircuitJSON format

```json
{
  "circuit_name": "and_2in",
  "description": "and gate with 2 input(s)",
  "circuit_type": "and",
  "nodes": [
    {"id": "in_0", "type": "input",  "name": "A0", "width": 1, "num_inputs": 2},
    {"id": "in_1", "type": "input",  "name": "A1", "width": 1, "num_inputs": 2},
    {"id": "gate0","type": "and",    "name": "AND0","width": 1, "num_inputs": 2},
    {"id": "out_0","type": "output", "name": "Y",  "width": 1, "num_inputs": 2}
  ],
  "edges": [
    {"from": "in_0", "to": "gate0", "from_port": 0, "to_port": 0, "name": "w_in0"},
    {"from": "in_1", "to": "gate0", "from_port": 0, "to_port": 1, "name": "w_in1"},
    {"from": "gate0","to": "out_0", "from_port": 0, "to_port": 0, "name": "w_y"}
  ]
}
```

---

## LLM integration

Set environment variables to use an OpenAI-compatible LLM:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # or GreenMind endpoint
```

If no API key is provided, the rule-based engine handles prompt parsing automatically.

---

## Verification

If [Icarus Verilog](https://github.com/steveicarus/iverilog) is installed:

```bash
sudo apt-get install iverilog   # Ubuntu/Debian
```

The verification module will compile and simulate the generated Verilog. Without iverilog, a pure-Python simulation is used as a fallback.