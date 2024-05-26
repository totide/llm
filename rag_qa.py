import os

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
llm = Ollama(model="llama3")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
Settings.llm = llm
Settings.embed_model = embed_model

loader = SimpleDirectoryReader(
    input_dir='./docs',
    required_exts=[".pdf"],
    recursive=True
)
docs = loader.load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(streaming=False, similarity_top_k=4)

qa_prompt_txt = """#OBJECTIVE#
根据我提供的query，我需要你一步一步回答我的问题
Query: {query_str}
#RESPONSE#
Answer
"""
qa_prompt_tmpl = PromptTemplate.from_template(qa_prompt_txt)
query_engine.update_prompts({"response_synthesizer: text_qa_template": qa_prompt_tmpl})

while True:
    query = input("请输入问题：")
    reponse = query_engine.query(query)
    print(reponse)
