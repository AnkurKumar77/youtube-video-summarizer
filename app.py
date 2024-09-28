import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit App

st.set_page_config(page_title="Langchain: Summarize Text From youtube or website. ")
st.title("Langchain: Summarize text from youtube or website. ")
st.subheader("Summarize URL")

## Get the groq api key

with st.sidebar:

    groq_api_key=st.text_input("Groq API Key", value="", type="password")
    
if groq_api_key:
    llm = ChatGroq(model="Gemma-7b-it", groq_api_key=groq_api_key)

    prompt_template="""
            Provide a summary of the following content in minimum 800 words:
            Content:{text}
            """
    prompt=PromptTemplate(template=prompt_template, input_variables=["text"])

    
url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize"):

    # validate the inputs
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the api key and url to get started.")
    elif not validators.url(url):
        st.error("Please enter a valid url.")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False)

                docs = loader.load()

                # Chain for summarization 
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")

