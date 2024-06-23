from typing import Union, List
from dotenv import load_dotenv
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import tool, Tool
from langchain.schema import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from callbacks import AgentCallbackHandler

load_dotenv()

# Note: Agent needs three things: llm, tools for it to use, and the instructions for it to follow (like you should pick a tool from provided ones only)
# Once you have these three details, you can start creating the agent


# @tool automatically converts a function into a langchain tool
@tool
# Writing a function to create a langchain tool out of it.
def interest_rate_calculator(months):
    """Returns the interest rate for the months provided"""
    # clean_text = "".join(char for char in text if char.isalpha())
    res = f"The interest rate for {months} months is 20%"
    return res

@tool
# Writing a function to create a langchain tool out of it.
def denomination_rate_calculator(months):
    """Returns the denomination rate for the months provided"""
    # clean_text = "".join(char for char in text if char.isalpha())
    # if you send the result as the "interest" rate is 23.88 (instead of denomination rate), this statement will be sent to llm. And llm thinks that although i asked to run denomination_rate, i am getting the interest_rate as answer. So this might have been incorrect. And then iterates with some another tool again without sending AgentFinish.
    res = f"The denomination rate for {months} months is 23.88%"
    return res


if __name__ == "__main__":
    # Making a list of tools that we want to provide to our agent (react agent)
    tools = [
        Tool(name="interest_rate_calculator", func=interest_rate_calculator,
             description="Returns the interest rate for the months provided"),
        Tool(name="denomination_rate_calculator", func=denomination_rate_calculator,
             description="Returns the denomination rate for the months provided")
    ]

    # tools = [{'name': get_text_length, 'description': 'This is Tool1.'}]
    # This template is going to be  sent to LLM to help it choose the right tool for the task
    template = """
    Answer the following questions as best you can. You have access to the following tools:
    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    # .partial() method is used to provide placeholders within the prompt template. We have 2 placeholders in our prompt (tools, tool_names), so declaring them
    # here, variable named 'tools' is a list. But, for llms we need to send only text, we have an inbuilt function in langchain to render list to text  i.e render_text_description
    prompt = PromptTemplate(template=template).partial(
        #render text to description attaches the description provided with the function with the function name while sending it to LLM. Tools are sent to llm along with their description like (interest_rate_calculator(months) - Returns the interest rate for the months provided, ....)
        tools=render_text_description(tools),
        tool_names=", ".join(t.name for t in tools),
    )

    def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
        for tool in tools:
            if tool.name== tool_name:
                return tool
        raise ValueError(f"Tool wtih name {tool_name} not found")

    # kwargs means keyword_arguments. model_kwargs is used to send extra arguments/initialization values into LLM
    # stop key word argument (stop:[...]) is used to stop llm from generating words/hallucinating after a tool is run and result of it is received.
    llm = ChatOpenAI(
        temperature=0,
        # we do not want the llm to send us it's result/answer directly to user question using its intelligence. It has to only suggest an action for langchain to run thatsall. So, we are asking the llm to stop as soon as the word observation (llms direct answer to question) is being generated.
        # We only want to generate the observation/result after running the tool, butnot the llms hallucination/it's direct answer using the dataset it was trained on to answer the user question directly.
        # So we want Action, Action input from the llm. After getting that we run that action with input using langchain runtime, and then in the next iteration to llm we append this result as an observation and share it to llm (for interpretation if that is final result or need more actions to be performed).
        # If the llm needs more steps to perform, the response from llm will be as an AgentAction (we get Action, Action_input using which we agian run the tool) or else we get an AgentFinish.
        # If response is an AgentFinish we terminate the iterations of sending observations to llm again, else we continue the loop as long as AgentFinish is received from llm
        stop=["\nObservation", "Observation"],
        callbacks=[AgentCallbackHandler()],
    )
    intermediate_steps = []

    # Note: Agent needs three things: llm, tools for it to use, and the instructions for it to follow (like you should pick a tool from provided ones only)
    # Once you have these three details, you can start creating the agent

    # agent takes the prompt and plugs it into the llm
    # Here, input is the question we ask to the agent. Based on this tool is chosen
    # Input is provided dynamically when invoking the agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    # As long as agent_step is not AgentFinish i.e we have something to do, this will go on runnning
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "Find the denomination rate for 12 months",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            # Running the function within that tool to get the result/observation for that chosen action. This will also be sent to llm in the next iteration
            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))

    # If agent step is agent finish, then print out the final return values
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
