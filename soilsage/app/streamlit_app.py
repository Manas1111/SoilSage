import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.rag import SoilSage
from src.soil_rules import classify, to_context

st.set_page_config(page_title="SoilSage", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge base...")
def load():
    return SoilSage(use_dense=os.getenv("SOILSAGE_DENSE", "1") != "0")


bot = load()
mode = bot.retriever.search("soil", k=1)[0]["mode"]

st.title("🌱 SoilSage")
st.caption("Soil health & crop advisory from ICAR/PAU/university documents, with page-level citations. "
           f"Retrieval mode: **{mode}**")

with st.sidebar:
    st.header("Soil test values (optional)")
    use = lambda label, **kw: st.number_input(label, value=None, placeholder="skip", **kw)
    ph = use("pH", min_value=3.0, max_value=11.0, step=0.1)
    ec = use("EC (dS/m)", min_value=0.0, max_value=30.0, step=0.1)
    oc = use("Organic carbon (%)", min_value=0.0, max_value=5.0, step=0.05)
    n = use("Available N (kg/ha)", min_value=0.0, step=10.0)
    p = use("Available P (kg/ha)", min_value=0.0, step=1.0)
    k = use("Available K (kg/ha)", min_value=0.0, step=10.0)
    crop = st.text_input("Crop (optional)", placeholder="e.g. wheat")

rows = classify(ph, ec, oc, n, p, k)
soil_ctx = to_context(rows)

if rows:
    st.subheader("Soil status")
    try:
        st.dataframe(rows, hide_index=True, width="stretch")
    except TypeError:  # older Streamlit
        st.dataframe(rows, hide_index=True, use_container_width=True)

default_q = ("What fertilizer and soil management do you recommend?" if rows
             else "")
question = st.text_input("Ask a soil or crop question",
                         value=default_q, placeholder="e.g. How to reclaim saline soil?")
if crop:
    question = f"{question} (crop: {crop})" if question else f"Recommendations for {crop}"

if st.button("Get advice", type="primary") and question:
    with st.spinner("Searching documents..."):
        res = bot.ask(question, soil_context=soil_ctx)
    if res["grounded"]:
        st.markdown(res["answer"])
    else:
        st.warning(res["answer"])
    if res["sources"]:
        st.subheader("Sources")
        for s in res["sources"]:
            with st.expander(f"[{s['n']}] {s['source']}, page {s['page']}"):
                st.write(s["snippet"] + "...")
                st.caption(f"retrieval score: {s['score']}")

st.divider()
st.caption("Advice is generated from the retrieved documents and is not a substitute for "
           "local agronomist or Soil Health Card recommendations.")
