from typing import Tuple
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain_community.llms import Ollama
from langchain.agents import AgentType, AgentExecutor, create_openai_functions_agent, create_tool_calling_agent
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langgraph.prebuilt import create_agent_executor
from langchain_openai import ChatOpenAI
import openai
import os
#from callbacks import AgentCallbackHandler
from tools.get_full_prompt import get_prompt
from tools.python_repl_plot_tool import PythonREPLPlotTool
from langchain.tools import Tool
from langchain import hub
import pickle
import logging
import traceback
import sys
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import logging
from pandas.tests.io.parser.conftest import csv_dir_path
from langchain.globals import set_debug
import glob
from flask import url_for

#set_debug(True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Get the absolute path of the current script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Go one directory up
root_directory = os.path.dirname(current_directory) + "/"

data_directory = root_directory + "static/data/"
csv_path = data_directory + "AmesHousing.csv"

sys.path.append(data_directory)

def generate_answer(question: str, llm_model: str) -> Tuple[str]:
    logging.info("Start generating answer...")
    try:
        
        load_dotenv()
       
        # Initialize llm using Ollama and llama3
        if ("ollama" == llm_model):
            llm = OllamaFunctions(model="llama3", temperature=0)
        # Initialize llm using OpenAI
        elif ("openai" == llm_model):
            llm = ChatOpenAI(temperature=0, model="gpt-4o")

        #callbacks=[AgentCallbackHandler()],

        # Define the enhanced prefix with plotting instructions
        enhanced_prefix = """
        You are an agent designed to write python code and invoke a Python REPL tool that can execute the python code that you generate.
        You can invoke it with the Action name "Python_REPL".
        This tool supports data access, manipulation, and visualization. Feel free to use it to generate plots.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question.
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        Do not simulate any data or files. You can safely assume that the Python REPL has access to all the files with data. 
        If you give any plotting instruction to the Python REPL, also give instruction to save the plot as a file. For example: plt.savefig('plot.svg')
        """

        
        python_agent_executor = create_python_agent(llm=llm, tool=PythonREPLPlotTool(data_directory=data_directory), agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                verbose=True, prefix=enhanced_prefix)
        #python_agent_executor.invoke("""How many rows are there in the enriched_mes.csv file?""")
        
        prompt = get_prompt(root_directory)
        print(prompt.format(input="", chat_history=[], agent_scratchpad=[]))

        tools = [
                Tool(
                    name="PythonAgent",
                    func=python_agent_executor.invoke,
                    description="""DO NOT SEND PYTHON CODE TO THIS TOOL DIRECTLY. Send only natural language queries. The agent has access to tools that can transform natural language into python code and execute it,
                                returning the results of the code execution. The tool also has access to all the files. So there is no need to simulate any file.
                                """,
                ),
            ]
        
        if ("ollama" == llm_model):
            # Bind the tools to the llm
            # llm = OllamaFunctions(model="llama3").bind_tools(tools=tools,
            # function_call={"PythonAgent": "python_agent_executor"})     
            grand_agent = create_openai_functions_agent(llm, tools, prompt)
            # Create the agent executor
            agent_executor = create_agent_executor(grand_agent, tools)
        elif ("openai" == llm_model):
            grand_agent = create_openai_functions_agent(
                tools=tools,
                llm=llm,
                prompt=prompt
            )
            # Create an agent executor by passing in the agent and tools
            agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

        #print(f"question = {question}")

        image_extensions = ['jpg', 'jpeg', 'png', 'svg']
        # If there are any image files, delete/clean them
        image_files = glob.glob(data_directory + '*.*')
        for image_file in image_files:
            if image_file.split('.')[-1] in image_extensions:
                os.remove(image_file)

        final_answer = "" 
        try:
            response = agent_executor.invoke({"input": question, "chat_history": []})
            # Check if the response contains an image
            final_answer = response['output']
        except Exception as e:
            # Log the error message and traceback
            logging.error(f"An error occurred: {e}")
            logging.error(traceback.format_exc())
            final_answer = "An error occurred while processing your request."
        else:
            # Log the Python agent's output
            python_agent_output = response.get('intermediate_steps', [])
            for step in python_agent_output:
                if step['tool'] == 'PythonAgent':
                    logging.info(f"Python Agent Output: {step['output']}")


        image_file_path = ""

        # Check if an image is generated

        # Find image files in the directory
        image_files = glob.glob('../static/data/*.*')
        print(f"All_files = {image_files}")
        for image_file in image_files:
            if image_file.split('.')[-1] in image_extensions:
                image_file_path = url_for('static', filename='data/' + os.path.basename(image_file))
        print(f"final_answer is = : {final_answer}, image_file_path is = : {image_file_path}")
        return final_answer, image_file_path

    except Exception as e:
        logging.error("An error occurred while processing the question.", exc_info=True)
        return "An error occurred while processing your request."


if __name__ == "__main__":
    question = """
    How many rows are there in the AmesHousing.csv file?
"""
    answer = generate_answer(question, "openai")
    print("Answer = ", answer)
