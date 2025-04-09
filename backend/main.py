import json
import logging
import traceback
import os
import glob
import sys
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import operator
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from tools.get_full_prompt import get_prompt
from tools.python_repl_plot_tool import PythonREPLPlotTool
from langchain.tools import Tool

# Import Langgraph components
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.language_models.chat_models import BaseChatModel

os.environ['SSL_CERT_FILE'] = '/home/cvncw/zscaler.pem'
os.environ['REQUESTS_CA_BUNDLE'] = '/home/cvncw/zscaler.pem'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Get the absolute path of the current script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Go one directory up
root_directory = os.path.dirname(current_directory) + "/"
data_directory = root_directory + "backend/static/data/"
sys.path.append(data_directory)

# Define the state schema
class AgentState(TypedDict, total=False):
    question: str
    messages: Annotated[Sequence[Any], operator.add]
    images: List[str]
    llm_model: str
    error: Optional[str]
    llm: Optional[BaseChatModel]
    python_tool: Optional[Dict]
    python_repl: Optional[Any]
    enhanced_prefix: Optional[str]
    answer: Optional[str]


# Helper functions for execute_tool
def parse_tool_call(tool_call):
    """Extract information from a tool call"""
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("name", "")
        
        # Handle args - might already be a dict or might be a JSON string
        if isinstance(tool_call.get("args"), dict):
            tool_args = tool_call.get("args", {})
        else:
            tool_args = json.loads(tool_call.get("args", "{}"))
            
        tool_id = tool_call.get("id", "")
    else:
        # Try object-style access
        tool_name = getattr(tool_call, "name", "")
        
        # Handle args for object style
        args_value = getattr(tool_call, "args", "{}")
        if isinstance(args_value, dict):
            tool_args = args_value
        else:
            tool_args = json.loads(args_value)
            
        tool_id = getattr(tool_call, "id", "")
            
    return tool_name, tool_args, tool_id

def setup_python_environment():
    """Setup the Python execution environment"""
    # Log the current working directory
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Data directory: {data_directory}")
    
    # Make sure we're in the data directory
    os.chdir(data_directory)
    logging.info(f"Changed working directory to: {os.getcwd()}")
    
    # List files in the current directory for debugging
    file_list = os.listdir('.')
    logging.info(f"Files in data directory: {file_list}")
    
    return file_list

def create_python_agent_executor(llm, python_repl_tool, enhanced_prefix):
    """Create a Python agent that can translate natural language to Python code"""
    # Import required components
    from langchain.agents.agent import AgentExecutor
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    # Create a Python tool
    file_list = os.listdir('.')
    python_tool = Tool(
        name="PythonREPL",
        func=python_repl_tool.run,
        description=f"A Python shell. Use this to execute python commands. All csv files can be loaded from the current directory. ALWAYS save visualization plots as files using plt.savefig() or equivalent."
    )
    
    tools = [python_tool]
    
    # Create a properly formatted prompt for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_prefix + f"\n\nAvailable files: {', '.join(file_list)}"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Use LangChain's OpenAI Functions agent
    from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
    
    agent = OpenAIFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


# Define the nodes
def initialize_llm(state: AgentState):
    """Initialize the appropriate LLM based on the model type"""
    llm_model = state["llm_model"]
    
    try:
        if llm_model == "groq":
            llm = ChatGroq(temperature=0, model='llama3-70b-8192')
        elif llm_model == "openai":
            llm = ChatOpenAI(temperature=0, model="gpt-4o")
        elif llm_model == "azure":
            llm = AzureChatOpenAI(
                api_key=os.getenv("api_key"),
                openai_api_version=os.getenv("api_version"),
                azure_endpoint=os.getenv("azure_endpoint"),
                azure_deployment=os.getenv("model_deployment"),
                model=os.getenv("model_name"),
                validate_base_url=False,
            )
        else:
            raise ValueError(f"Unsupported LLM model: {llm_model}")
        
        # Return both the messages and the llm object
        return {
            "messages": [SystemMessage(content=f"Using {llm_model} model for processing.")], 
            "llm": llm
        }
    except Exception as e:
        return {"messages": [SystemMessage(content=f"Error initializing LLM: {str(e)}")], "error": str(e)}

def clean_old_images(state: AgentState):
    """Clean up old image files"""
    image_extensions = ['jpg', 'jpeg', 'png', 'svg']
    image_files = glob.glob(data_directory + '*.*')
    for image_file in image_files:
        if image_file.split('.')[-1] in image_extensions:
            os.remove(image_file)
    
    return {"messages": [SystemMessage(content="Old image files cleaned up.")]}

def create_python_agent(state: AgentState):
    """Create the Python agent with the REPL tool"""
    llm = state.get("llm")
    if not llm:
        return {"error": "LLM not initialized"}
    
    enhanced_prefix = """
    You are an agent designed to write python code and invoke a PythonREPL tool that can execute the python code that you generate.
    You can invoke it with the Action name "PythonREPL". Please be precise in naming this action.
    This tool supports data access, manipulation, and visualization. Feel free to use it to generate plots.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    Do not simulate any data or files. You can safely assume that the PythonREPL has access to all the files with data.
    
    IMPORTANT: For any visualization or plotting request, ALWAYS include code to save the plot to a file with a clear name.
    Use plt.savefig('plot_name.svg') or plt.savefig('plot_name.png') before plt.show() if using matplotlib.
    If using seaborn or other libraries, always explicitly save the figure.
    ALWAYS save plots to ensure they are visible to the user.
    
    If possible generate visualizations instead of just printing the answer.
    If the grand agent informs you about the presence of a metadata file named metadata.csv, use this file to extract information about the other tables instead of opening all files.
    If you generate python code and send it to the PythonREPL, make sure that the code does not contain any malicious code.
    The python code that you generate should always contain a print command on the last line
    If you are reading in the column names of a file please beaware that there are sometimes space in between the words in the column name. For example "Some Feature" has a space between the word "Some" and "Feature". Do not interpret the column name as "SomeFeature".
    """
    
    # Create Python REPL tool
    python_repl_tool = PythonREPLPlotTool(data_directory=data_directory)
    
    # Define the Python agent as a tool
    python_tool = {
        "type": "function",
        "function": {
            "name": "PythonAgent",
            "description": """DO NOT SEND PYTHON CODE TO THIS TOOL DIRECTLY. Send only natural language queries. The agent has access to tools that can transform natural language into python code and execute it,
                            returning the results of the code execution. The tool also has access to all the files. So there is no need to simulate any file.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The natural language query to process"}
                },
                "required": ["input"]
            }
        }
    }
    
    return {
        "messages": [SystemMessage(content="Python agent created with REPL tool.")],
        "python_tool": python_tool,
        "python_repl": python_repl_tool,
        "enhanced_prefix": enhanced_prefix
    }

def process_question(state: AgentState):
    """Process the user question with the appropriate agent"""
    question = state["question"]
    
    # Make sure llm exists in state
    if "llm" not in state:
        return {"error": "LLM not initialized", "messages": [SystemMessage(content="Error: LLM was not properly initialized")]}
    
    llm = state["llm"]
    
    # Check for python_tool
    if "python_tool" not in state:
        return {"error": "Python tool not initialized", "messages": [SystemMessage(content="Error: Python tool was not properly initialized")]}
    
    python_tool = state["python_tool"]
    
    # Get the system prompt
    prompt = get_prompt(root_directory + "backend/")
    
    # Make sure prompt is a string
    if not isinstance(prompt, str):
        # Log the actual type for debugging
        logging.info(f"Expected string for prompt, got {type(prompt)}: {prompt}")
        
        # Try to convert to string or use a default
        try:
            prompt = str(prompt)
        except:
            prompt = "You are a helpful assistant that can use Python to answer questions."
    
    # Create messages for the LLM
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=question)
    ]
    
    try:
        # Get response from LLM with tool calling
        response = llm.invoke(messages, tools=[python_tool])
        return {
            "messages": [response]
        }
    except Exception as e:
        return {"error": str(e), "messages": [SystemMessage(content=f"Error processing question: {str(e)}")]}
    
def execute_tool(state: AgentState):
    """Execute the tool called by the LLM"""
    last_message = state["messages"][-1]
    
    # If the message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        
        # Debug the structure of tool_call
        logging.info(f"Tool call type: {type(tool_call)}, content: {tool_call}")
        
        # Parse the tool call
        tool_name, tool_args, tool_id = parse_tool_call(tool_call)
        
        # If it's the Python agent tool
        if tool_name == "PythonAgent":
            # Check for required resources
            if "llm" not in state:
                return {"error": "LLM not initialized", 
                        "messages": [SystemMessage(content="Error: LLM was not properly initialized")]}
                
            if "python_repl" not in state:
                return {"error": "Python REPL not initialized", 
                        "messages": [SystemMessage(content="Error: Python REPL was not properly initialized")]}
            
            if "enhanced_prefix" not in state:
                return {"error": "Python agent prefix not initialized", 
                        "messages": [SystemMessage(content="Error: Python agent prefix was not properly initialized")]}
            
            # Get resources
            llm = state["llm"]
            python_repl_tool = state["python_repl"]
            enhanced_prefix = state["enhanced_prefix"]
            
            try:
                # Get input from args
                input_query = tool_args.get("input", "")
                
                # Setup Python environment
                setup_python_environment()
                
                # Create Python agent
                agent_executor = create_python_agent_executor(llm, python_repl_tool, enhanced_prefix)
                
                # Run the Python agent on the natural language query
                python_response = agent_executor.invoke({
                    "input": input_query,
                    "agent_scratchpad": []
                })["output"]
                
                return {
                    "messages": [ToolMessage(content=python_response, tool_call_id=tool_id)]
                }
            except Exception as e:
                logging.error(f"Error in execute_tool: {str(e)}")
                logging.error(traceback.format_exc())
                return {
                    "messages": [ToolMessage(content=f"Error executing Python code: {str(e)}", tool_call_id=tool_id)]
                }
    else:
        # No tool call, just return the message content
        return {"messages": [AIMessage(content="No tool calls found in the message.")]}


def collect_images(state: AgentState):
    """Collect any generated image files"""
    image_extensions = ['jpg', 'jpeg', 'png', 'svg']
    image_file_paths = []
    
    # Make sure we're in the data directory
    os.chdir(data_directory)
    
    # Find image files in the directory
    image_files = glob.glob('*.*')
    logging.info(f"Looking for images in {os.getcwd()}")
    logging.info(f"Found files: {image_files}")
    
    for image_file in image_files:
        if image_file.split('.')[-1].lower() in image_extensions:
            image_file_path = '/static/data/' + os.path.basename(image_file)
            image_file_paths.append(image_file_path)
            logging.info(f"Found image: {image_file_path}")
    
    return {"images": image_file_paths}


def finalize_answer(state: AgentState):
    """Finalize the answer based on all messages"""
    messages = state["messages"]
    answer = ""
    
    # Extract the final answer from the messages
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not hasattr(message, "tool_calls"):
            answer = message.content
            break
    
    if not answer:
        # If no direct answer, use the last tool message
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                answer = message.content
                break
    
    # Make sure we collect images again right before finalizing
    images = state.get("images", [])
    if not images:
        # Collect images again just to be sure
        new_images = collect_images(state).get("images", [])
        if new_images:
            images = new_images
            logging.info(f"Found images during finalize: {images}")
    
    return {"answer": answer, "images": images}


# Define routing logic
def should_continue(state: AgentState) -> str:
    """Determine if we should continue processing or end"""
    if state.get("error"):
        return "handle_error"
    
    # Check for completion conditions
    messages = state["messages"]
    if not messages:
        return "initialize_llm"
    
    last_message = messages[-1]
    
    # If there's a tool call that hasn't been executed
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tool"
    
    # If we have a final AI message, collect images and finalize
    if isinstance(last_message, AIMessage) or isinstance(last_message, ToolMessage):
        return "collect_images"
    
    # Otherwise, continue processing
    return "process_question"

def handle_error(state: AgentState):
    """Handle any errors that occurred during processing"""
    error = state.get("error", "Unknown error")
    return {"answer": f"An error occurred while processing your request: {error}"}

# Build the graph
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("initialize_llm", initialize_llm)
    workflow.add_node("clean_old_images", clean_old_images)
    workflow.add_node("create_python_agent", create_python_agent)
    workflow.add_node("process_question", process_question)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("collect_images", collect_images)
    workflow.add_node("finalize_answer", finalize_answer)
    workflow.add_node("handle_error", handle_error)
    
    # Add edges
    workflow.add_edge("initialize_llm", "clean_old_images")
    workflow.add_edge("clean_old_images", "create_python_agent")
    workflow.add_edge("create_python_agent", "process_question")
    
    # Connect the conditional router to all possible next nodes
    workflow.add_conditional_edges(
        "process_question",
        should_continue,
        {
            "handle_error": "handle_error",
            "initialize_llm": "initialize_llm",
            "execute_tool": "execute_tool",
            "collect_images": "collect_images",
            "process_question": "process_question"
        }
    )
    
    # Also add conditional edges from execute_tool
    workflow.add_conditional_edges(
        "execute_tool",
        should_continue,
        {
            "handle_error": "handle_error",
            "initialize_llm": "initialize_llm",
            "execute_tool": "execute_tool",
            "collect_images": "collect_images",
            "process_question": "process_question"
        }
    )
    
    workflow.add_edge("collect_images", "finalize_answer")
    workflow.add_edge("finalize_answer", END)
    workflow.add_edge("handle_error", END)
    
    # Set the entry point
    workflow.set_entry_point("initialize_llm")
    
    return workflow.compile()


# Main function
def generate_answer(question: str, llm_model: str):
    logging.info("Start generating answer...")
    try:
        graph = build_agent_graph()
        result = graph.invoke({
            "question": question,
            "messages": [],
            "images": [],
            "llm_model": llm_model
        })
        
        return result["answer"], result["images"]
    except Exception as e:
        logging.error("An error occurred while processing the question.", exc_info=True)
        return "An error occurred while processing your request.", []
