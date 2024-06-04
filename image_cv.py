import base64
from io import BytesIO

from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser


llm = ChatOllama(model="ghyghoo8/minicpm-llama3-2_5:8b", temperature=0.5)


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    content_parts = [
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{data['image']}",
        },
        {"type": "text", "text": data["text"]}
    ]
    return [HumanMessage(content=content_parts)]


file_path = "static/img/ticket5.jpeg"
pil_image = Image.open(file_path)

image_b64 = convert_to_base64(pil_image)
chain = prompt_func | llm | StrOutputParser()
query_chain = chain.invoke(
    {"text": "请读取图片中的店铺名、交易时间、交易号、实付金额, 用json格式输出", "image": image_b64}
)
print(query_chain)
