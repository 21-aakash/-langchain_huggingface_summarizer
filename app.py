import os  # Import os to interact with the operating system
import validators  # Import validators to validate URLs
import streamlit as st  # Import Streamlit for building the web app
from dotenv import load_dotenv  # Import dotenv to load environment variables
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from the .env file
load_dotenv()

# Custom CSS for the neon effect
st.markdown("""
    <style>
    .neon-title {
        color: #00FF00;  /* Bright neon green */
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        text-shadow: 0 0 5px #00FF00, 0 0 10px #00FF00, 0 0 15px #00FF00, 0 0 20px #00FF00, 0 0 25px #00FF00, 0 0 30px #00FF00, 0 0 35px #00FF00;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app settings
st.set_page_config(page_title="SkyChat: Summarize Text From YT or Website", page_icon="👽")
st.markdown('<h1 class="neon-title">👽Skychat: Summarize Text From YT or Website</h1>', unsafe_allow_html=True)
st.subheader('Summarize the URL')

# Get the Hugging Face API key and URL from the user
with st.sidebar:
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "")  # Fetch API key from environment variables
    if not hf_api_key:
        hf_api_key = st.text_input("Huggingface API Token", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Set up the Hugging Face model
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_api_key)

# Define the prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    # Validate all inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load content from the URL
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()

                # Create the summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
