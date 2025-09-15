import streamlit as st
import langchain_helper

st.title("Souled Store Tshirts Database")

question = st.text_input("Question:")

if question:

    chain = langchain_helper.db_chain()
    result = chain.run(question)

    st.title("Answer:")
    st.subheader(result)