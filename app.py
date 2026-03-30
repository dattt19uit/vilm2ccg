"""
app.py – ViLM2CCG Streamlit Dashboard

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import streamlit as st

from vilm2ccg.llm_interface import LLMInterface, build_circuit_from_intent
from vilm2ccg.input_module import parse_vietnamese_prompt
from vilm2ccg.visualizer import visualize_circuit
from vilm2ccg.hdl_generator import generate_verilog
from vilm2ccg.verification import run_verification, compute_truth_table

_MODELS_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ViLM2CCG – Vietnamese LM to Circuit Graph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Cài đặt / Settings")
    st.markdown("---")

    st.markdown("#### 🔌 Backend")
    backend = st.radio(
        "Chọn backend / Select backend:",
        options=["Rule-based", "Local Model (seq2seq)", "LM Studio"],
        index=0,
        help=(
            "Rule-based: dùng engine tổ hợp xác định.\n"
            "Local Model: dùng model seq2seq đã train từ models/.\n"
            "LM Studio: gửi prompt tới LM Studio server."
        ),
    )

    # -- Local Model settings -----------------------------------------------
    local_model_dir = None
    if backend == "Local Model (seq2seq)":
        st.markdown("#### 🧠 Local Model")
        available_models: list[str] = []
        if _MODELS_DIR.exists():
            available_models = [
                d.name
                for d in sorted(_MODELS_DIR.iterdir())
                if d.is_dir() and (d / "model.pt").exists()
            ]
        if available_models:
            selected_model = st.selectbox(
                "Chọn checkpoint / Select checkpoint:",
                options=available_models,
                help=f"Các checkpoint trong thư mục {_MODELS_DIR}",
            )
            local_model_dir = str(_MODELS_DIR / selected_model)
            st.success(f"✅ Checkpoint: `{selected_model}`")
        else:
            st.warning(
                f"⚠️ Chưa có model trong `models/`. "
                "Hãy train trước:\n```\npython train.py --epochs 1\n```"
            )
            local_model_dir = str(_MODELS_DIR / "vilm2ccg-t5")

    # -- LM Studio settings -------------------------------------------------
    lm_studio_url = ""
    lm_studio_model = ""
    if backend == "LM Studio":
        st.markdown("#### 🖥️ LM Studio")
        lm_studio_url = st.text_input(
            "LM Studio URL",
            value=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
            help="Địa chỉ server LM Studio. Mặc định: http://localhost:1234/v1",
        )
        lm_studio_model = st.text_input(
            "Model name",
            value=os.getenv("LM_STUDIO_MODEL", "local-model"),
            help=(
                "Tên model đang chạy trong LM Studio. "
                "Dùng 'local-model' nếu chỉ có một model được load."
            ),
        )

    st.markdown("---")
    st.markdown("**Ví dụ prompts / Example prompts:**")
    examples = [
        "Tạo mạch MUX 4-1",
        "Mạch AND 3 đầu vào",
        "Cổng XOR 2 đầu vào",
        "Bộ cộng nửa",
        "Bộ cộng đầy đủ",
        "Mạch NOR 2 đầu vào",
        "Bộ giải mã 2 đầu vào",
        "Bộ so sánh 1 bit",
        "Cổng NOT",
        "Mạch OR 4 đầu vào",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state["prompt"] = ex

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("⚡ ViLM2CCG")
st.subheader("Vietnamese Language Model → Combinational Circuit Graph")
st.markdown(
    "Nhập mô tả mạch tổ hợp bằng **tiếng Việt** để sinh sơ đồ mạch, "
    "mã Verilog và bảng chân lý."
)

prompt_value = st.session_state.get("prompt", "Tạo mạch MUX 4-1")
prompt = st.text_area(
    "📝 Mô tả mạch (tiếng Việt) / Circuit description (Vietnamese):",
    value=prompt_value,
    height=80,
    key="prompt_input",
)

run_btn = st.button("🚀 Tạo mạch / Generate Circuit", type="primary")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

if run_btn and prompt.strip():
    with st.spinner("Đang xử lý… / Processing…"):
        # 1. Parse intent
        intent = parse_vietnamese_prompt(prompt)

        # 2. Build circuit based on selected backend
        _model_source: str = "rule-based"
        if backend == "Local Model (seq2seq)":
            from inference import ViLM2CCGInference
            from vilm2ccg.circuit_json import CircuitJSON
            engine = ViLM2CCGInference(model_dir=local_model_dir)
            circuit_dict = engine.generate_circuit_dict(prompt)
            circuit = CircuitJSON.from_dict(circuit_dict)
            _model_source = engine.last_source  # "model" or "fallback"
        elif backend == "LM Studio":
            llm = LLMInterface(
                base_url=lm_studio_url,
                model=lm_studio_model,
                use_fallback=False,
            )
            circuit = llm.generate_circuit(prompt)
        else:
            llm = LLMInterface(use_fallback=True)
            circuit = llm.generate_circuit(prompt)

        # 3. Generate Verilog
        verilog_code = generate_verilog(circuit)

        # 4. Visualize
        img_bytes = visualize_circuit(circuit, fmt="png")

        # 5. Verification
        try:
            ver_result = run_verification(circuit)
        except Exception as _ver_exc:
            from vilm2ccg.verification import VerificationResult
            ver_result = VerificationResult(
                passed=False,
                errors=-1,
                stdout="",
                stderr=str(_ver_exc),
                iverilog_available=False,
            )

        # 6. Truth table
        try:
            truth_table = compute_truth_table(circuit)
        except Exception:
            truth_table = []

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    st.success(f"✅ Đã tạo mạch: **{circuit.circuit_name}**")
    if backend == "Local Model (seq2seq)":
        if _model_source == "model":
            st.info("🤖 Kết quả từ seq2seq model.")
        else:
            st.warning(
                "⚠️ Model chưa sinh được JSON hợp lệ (cần train thêm epoch). "
                "Mạch được tạo bằng rule-based fallback.\n\n"
                "Để cải thiện: `python train.py --epochs 10`"
            )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔍 Phân tích / Analysis", "🖼️ Sơ đồ / Graph", "💾 Verilog", "📊 Bảng chân lý / Truth Table", "✅ Kiểm tra / Verification"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Kết quả phân tích / Parse Result")
            st.json({
                "circuit_type": intent.circuit_type,
                "num_inputs": intent.num_inputs,
                "num_select": intent.num_select,
                "data_width": intent.data_width,
                "circuit_name": intent.circuit_name,
            })
        with col2:
            st.markdown("### 📋 CircuitJSON")
            st.json(circuit.to_dict())

    with tab2:
        st.markdown("### 🖼️ Circuit Graph")
        st.image(img_bytes, use_column_width=True)
        st.download_button(
            "⬇️ Tải ảnh PNG / Download PNG",
            data=img_bytes,
            file_name=f"{circuit.circuit_name}.png",
            mime="image/png",
        )

    with tab3:
        st.markdown("### 💾 Verilog Code")
        st.code(verilog_code, language="verilog")
        st.download_button(
            "⬇️ Tải file Verilog / Download Verilog",
            data=verilog_code,
            file_name=f"{circuit.circuit_name}.v",
            mime="text/plain",
        )

    with tab4:
        st.markdown("### 📊 Truth Table")
        if truth_table:
            st.dataframe(truth_table, use_container_width=True)
        else:
            st.info("Không thể tính bảng chân lý cho mạch này.")

    with tab5:
        st.markdown("### ✅ Verification")
        if ver_result.iverilog_available:
            st.info("🔧 Đang sử dụng **Icarus Verilog** để mô phỏng.")
        else:
            st.warning(
                "⚠️ Icarus Verilog không được cài đặt. "
                "Đang sử dụng Python simulation fallback."
            )

        if ver_result.passed:
            st.success("✅ Mạch hoạt động đúng! / Circuit verified successfully!")
        else:
            st.error(f"❌ Phát hiện {ver_result.errors} lỗi / Found {ver_result.errors} error(s)!")

        st.text_area("Simulation output:", value=ver_result.stdout, height=200)
        if ver_result.stderr:
            st.text_area("Stderr:", value=ver_result.stderr, height=100)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:12px;'>"
    "ViLM2CCG v0.1.0 — Vietnamese LM to Combinational Circuit Graph<br/>"
    "Powered by NetworkX, Matplotlib, Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
