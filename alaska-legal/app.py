"""
app.py

Streamlit UI for the Alaska Administrative Code section lookup system.
Single-page, single-column. Calls retriever and answerer only.

Usage:
    streamlit run app.py
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from retriever import retrieve
from answerer import answer

st.title("Alaska Administrative Code — Section Lookup")
st.caption("Enter your query including a section number in format 3 AAC XX.XXX")

query = st.text_input("Query")

if st.button("Look up section"):
    if not query.strip():
        st.error("Please enter a query.")
        st.stop()

    with st.spinner("Looking up section..."):
        result = retrieve(query)

        if isinstance(result, dict) and "error" in result:
            st.error(result["error"])
            st.stop()

        response = answer(result, query)

    st.markdown(f"**3 AAC {result.section_id} — {result.title}**")

    if result.status == "repealed":
        st.warning(response)
        st.stop()

    st.divider()
    st.markdown(response)

    with st.expander("View full section text", expanded=False):
        st.text(result.text)
