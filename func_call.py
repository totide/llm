from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, load_tools
from langchain_community.chat_models import ChatOllama
from tools.Calculator import Calculator


llm = ChatOllama(model="llama3")
prompt = hub.pull("hwchase17/structured-chat-agent")


def _math_calc():
    tools = [Calculator()]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    ans = agent_executor.invoke({"input": "34 * 34"})
    print(ans)


def _builtin_tool():
    tools = load_tools(["arxiv"], llm=llm)
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    ans = agent_executor.invoke({"input": "Describe the paper about GLM 130B"})
    print(ans)


if __name__ == '__main__':
    _math_calc()
    # _builtin_tool()
