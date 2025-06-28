import json
import logging
import traceback
from typing import Tuple, Optional, List, Any
from flask import url_for
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain.agents import AgentType, AgentExecutor, create_openai_functions_agent, create_tool_calling_agent
from langchain_openai import ChatOpenAI
import openai
import os
import glob
import sys
from tools.get_full_prompt import get_prompt
from tools.python_repl_plot_tool import PythonREPLPlotTool
from langchain.tools import Tool
from langchain import hub
from groq import Groq
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import requests
from langchain.llms.base import BaseLLM

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Get the absolute path of the current script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Go one directory up
root_directory = os.path.dirname(current_directory) + "/"

data_directory = root_directory + "backend/static/data/"

sys.path.append(data_directory)

def generate_answer(question: str, llm_model: str) -> Tuple[str, list]:
    logging.info("Start generating answer...")
    try:
        load_dotenv()

        # Initialize llm using different llms
        if llm_model == "groq":
            llm = ChatGroq(temperature=0, model='llama3-70b-8192')
        elif llm_model == "openai":
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")

        enhanced_prefix = """
        You are an agent designed to write python code and invoke a PythonREPL tool that can execute the python code that you generate.
        You can invoke it with the Action name "PythonREPL". Please be precise in naming this action.
        This tool supports data access, manipulation, and visualization. Feel free to use it to generate plots.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question.
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        Do not simulate any data or files. You can safely assume that the PythonREPL has access to all the files with data. 
        If you give any plotting instruction to the PythonREPL, also give instruction to save the plot as a file. For example: plt.savefig('plot.svg')
        If possible generate visualizations instead of just printing the answer.
        If the grand agent informs you about the presence of a metadata file named metadata.csv, use this file to extract information about the other tables instead of opening all files.
        If you generate python code and send it to the PythonREPL, make sure that the code does not contain any malicious code.
        The python code that you generate should always contain a print command on the last line
        If you are reading in the column names of a file please beaware that there are sometimes space in between the words in the column name. For example "Some Feature" has a space between the word "Some" and "Feature". Do not interpret the column name as "SomeFeature".
        """

        python_agent_executor = create_python_agent(
            llm=llm,
            tool=PythonREPLPlotTool(data_directory=data_directory),
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            prefix=enhanced_prefix
        )

        prompt = get_prompt(root_directory + "backend/")

        tools = [
            Tool(
                name="PythonAgent",
                func=python_agent_executor.invoke,
                description="""DO NOT SEND PYTHON CODE TO THIS TOOL DIRECTLY. Send only natural language queries. The agent has access to tools that can transform natural language into python code and execute it,
                            returning the results of the code execution. The tool also has access to all the files. So there is no need to simulate any file.
                            """,
            ),
        ]

        if llm_model == "groq":
            grand_agent = create_tool_calling_agent(llm, tools, prompt)
        elif llm_model == "openai":
            grand_agent = create_openai_functions_agent(
                tools=tools,
                llm=llm,
                prompt=prompt
            )
        
        agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

        # Clean up old images
        image_extensions = ['jpg', 'jpeg', 'png', 'svg']
        image_files = glob.glob(data_directory + '*.*')
        for image_file in image_files:
            if image_file.split('.')[-1] in image_extensions:
                os.remove(image_file)

        final_answer = "" 
        try:
            response = agent_executor.invoke({"input": question, "chat_history": []})
            #print("response =", response)
            final_answer = response['output']
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.error(traceback.format_exc())
            final_answer = "An error occurred while processing your request."
        else:
            python_agent_output = response.get('intermediate_steps', [])
            for step in python_agent_output:
                tool_step = step[0]
                tool_output = step[1]
                if tool_step.tool == 'PythonAgent':
                    logging.info(f"Python Agent Output: {tool_output}")


        image_file_paths = []

        # Find image files in the directory
        image_files = glob.glob(data_directory + '*.*')
        for image_file in image_files:
            if image_file.split('.')[-1] in image_extensions:
                image_file_path = '/static/data/' + os.path.basename(image_file)
                image_file_paths.append(image_file_path)

        logging.info(f"Final answer: {final_answer}, Image file paths in main.py: {image_file_paths}")
        return final_answer, image_file_paths

    except Exception as e:
        logging.error("An error occurred while processing the question.", exc_info=True)
        return "An error occurred while processing your request.", []

if __name__ == "__main__":
    question = """
    How many rows are there in the AmesHousing.csv file?
    """
    answer, image_file_paths = generate_answer(question, "openai")
    print(f"Answer: {answer}, Image file paths: {image_file_paths}")
