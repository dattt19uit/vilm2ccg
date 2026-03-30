"""
app.py – ViLM2CCG Streamlit Dashboard

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os

import streamlit as st

from vilm2ccg.llm_interface import LLMInterface, build_circuit_from_intent
from vilm2ccg.input_module import parse_vietnamese_prompt
from vilm2ccg.visualizer import visualize_circuit
from vilm2ccg.hdl_generator import generate_verilog
from vilm2ccg.verification import run_verification, compute_truth_table

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

    api_key = st.text_input(
        "OpenAI API Key (tuỳ chọn / optional)",
        type="password",
        help="Để trống để sử dụng rule-based engine.",
    )
    model_choice = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "greenmind"],
        index=0,
    )
    use_llm = st.checkbox("Sử dụng LLM", value=bool(api_key))

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

        # 2. Build circuit (LLM or rule-based)
        llm = LLMInterface(
            api_key=api_key if use_llm else "",
            model=model_choice,
            use_fallback=not use_llm,
        )
        circuit = llm.generate_circuit(prompt)

        # 3. Generate Verilog
        verilog_code = generate_verilog(circuit)

        # 4. Visualize
        img_bytes = visualize_circuit(circuit, fmt="png")

        # 5. Verification
        ver_result = run_verification(circuit)

        # 6. Truth table
        truth_table = compute_truth_table(circuit)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    st.success(f"✅ Đã tạo mạch: **{circuit.circuit_name}**")

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
