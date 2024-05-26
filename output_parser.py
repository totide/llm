from langchain.llms.ollama import Ollama
from langchain.output_parsers import RetryOutputParser, ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel


class Joke(BaseModel):
    question: str = Field(description="question to set up a joke")
    answer: str = Field(description="answer to resolve the joke")


parser = JsonOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
llm = Ollama(model="llama3", )


def not_try_prompt():
    chain = prompt | llm | parser
    print(chain.invoke({"query": "Tell me a joke"}))


def err_try_prompt():
    retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm)

    completion_chain = prompt | llm
    main_chain = RunnableParallel(
        completion=completion_chain, prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    print(main_chain.invoke({"query": "Tell me a joke."}))


def struct_output(question="who is leo di caprios gf?"):
    """
    该方法用于将输出格式化为结构化数据。它使用 ResponseSchema 对象来定义要解析的输出，并通过 StructuredOutputParser 对象来实现格式化。
    """
    response_schemas = [
        ResponseSchema(name="answer", description="answer to the user's question"),
        ResponseSchema(
            name="source",
            description="source used to answer the user's question, should be a website.",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    _prompt = PromptTemplate(
        template="answer the users question as best as possible.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )
    chain = _prompt | llm | output_parser
    # print(chain.invoke({"question": "what's the capital of france?"}))
    print(chain.invoke({"question": question}))


if __name__ == '__main__':
    # not_try_prompt()
    err_try_prompt()
    struct_output()
