import os
import re
import fitz
from langchain.llms.ollama import Ollama
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from llama_index.core.readers import StringIterableReader
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.settings import Settings

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class PDFChat(object):
    CHAPTER_REG = re.compile(r'^\d+ \n')

    def __init__(self, model_name="llama3", embed_model_name="BAAI/bge-m3"):
        self.llm = Ollama(model=model_name)
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

    @staticmethod
    def extract_pdf_content(pdf_path):
        """# PDF内容提取（分页处理）"""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                yield page.get_text()

    @staticmethod
    def build_qa_index(text_stream):
        """构建问答索引（逐页处理）"""
        documents = [text for text in text_stream]
        documents = StringIterableReader().load_data(texts=documents)
        index = VectorStoreIndex.from_documents(documents)
        return index

    @staticmethod
    def is_chapter_page(text):
        """FIXME: 这里需要一个识别章节标题，例如使用正则表达式, 或者通过目录获取相应的页码"""
        return text.startswith("Chapter") or PDFChat.CHAPTER_REG.search(text[:20])

    def generate_summary(self, text):
        """生成摘要"""
        print(f'generating summary...')
        return self.llm.invoke(f"Please summarize the following text: {text}\nSummary:")

    def generate_chapter_summaries(self, text_stream):
        """# 按章节标题提取文本内容并生成摘要"""
        summaries = []
        current_chapter = ""
        for text in text_stream:
            if self.is_chapter_page(text):
                if current_chapter:
                    chapter_summary = self.generate_summary(current_chapter)
                    summaries.append(chapter_summary)

                current_chapter = text
            else:
                current_chapter += text

        if current_chapter:
            chapter_summary = self.generate_summary(current_chapter)
            summaries.append(chapter_summary)
        return summaries

    @staticmethod
    def interactive_qa(qa_index):
        qa_prompt_txt = "Answer the question: {query_str}\nAnswer:"
        qa_prompt_tmpl = PromptTemplate.from_template(qa_prompt_txt)
        query_engine = qa_index.as_query_engine(streaming=False, similarity_top_k=4)
        query_engine.update_prompts({"response_synthesizer: text_qa_template": qa_prompt_tmpl})
        while True:
            user_query = input("请提出你的问题: ")
            if user_query.lower() == 'exit':
                break

            response = query_engine.query(user_query)
            print(f"AI助手：{response}")

    def print_chapter_summaries(self, path):
        """按章节标题生成摘要"""
        pdf_content_stream = self.extract_pdf_content(path)
        chapter_summaries = self.generate_chapter_summaries(pdf_content_stream)
        print("生成的章节摘要信息:\n")
        for summary in chapter_summaries:
            print(summary)

    def run_interactive_qa(self, path):
        """交互问答"""
        pdf_content_stream = self.extract_pdf_content(path)
        qa_index = self.build_qa_index(pdf_content_stream)
        self.interactive_qa(qa_index)


if __name__ == '__main__':
    pdf_path = 'docs/handbook.pdf'
    chat = PDFChat()
    chat.print_chapter_summaries(pdf_path)
    chat.run_interactive_qa(pdf_path)
