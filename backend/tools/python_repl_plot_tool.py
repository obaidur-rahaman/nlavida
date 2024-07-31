import ast
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Type

from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.tools.base import BaseTool
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor

from langchain_experimental.utilities.python import PythonREPL
from contextlib import contextmanager
import os
import logging
import base64
import glob 

def _get_default_python_repl() -> PythonREPL:
    return PythonREPL(_globals=globals(), _locals=None)

def add_matplotlib_use_agg(query):
    # Find all lines that import matplotlib
    matplotlib_import_lines = re.findall(r"(.*matplotlib.*)", query)

    for line in matplotlib_import_lines:
        # Add "matplotlib.use('Agg')" immediately after each import line
        replacement_line = line + "\nimport matplotlib\nmatplotlib.use('Agg')"
        query = query.replace(line, replacement_line)

    return query

def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.

    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)

    # Remove any text after the actual Python code and lines starting with Thought:
    query = re.split(r'\n?```\n?', query)[0].strip()
    query = re.sub(r'^Thought:.*$', '', query, flags=re.MULTILINE)

    # The code will also fail if it tries to show a plot.
    # Let's handle that sitution as well

    if "import matplotlib" in query: # In case plotting is involved
        # Add a line "matplotlib.use('Agg')" immediately after if "import matplotlib" is detected
        query = add_matplotlib_use_agg(query)
        # Remove any lines that contain "plt.show()"
        query = re.sub(r"plt.show\(\)", "", query)
        query = query + "\nprint(\"Plot is created successfully.\")\n"
    else: # In case plotting is not involved
        query = re.sub(r'\\(?!n)', '', query)
      
    print(f"\n\nSanitized query = \n\n{query}\n\n")
    return query

@contextmanager
def change_dir(target_directory):
    """Context manager to temporarily change the working directory."""
    current_dir = os.getcwd()
    try:
        os.chdir(target_directory)
        yield
    finally:
        os.chdir(current_dir)

class PythonREPLPlotTool(BaseTool):
    """Tool for running python code in a REPL, with access to a specific data directory, with enhanced capabilities for data access and plotting.
    Use this tool to execute python commands, especially for data manipulation and visualization.
    Ensure that input commands are properly sanitized. This tool supports dynamic plotting commands and can save plots as needed."""

    name: str = "PythonREPL"
    description: str = (
        "A Python shell enhanced for data access and plotting. Use this tool "
        "to execute python commands, especially for data manipulation and visualization. "
        "Ensure that input commands are properly sanitized."
    )
    data_directory: str = Field(default="")
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True

    def __init__(self, data_directory: str, **kwargs):
        super().__init__(**kwargs)
        self.data_directory = data_directory  # Initialize the data directory

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
        """Execute the tool synchronously, with error handling."""
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
                query = query.replace('\\n', '\n')
            with change_dir(self.data_directory):               
                result = self.python_repl.run(query)
                logging.debug(f"Output from Python agent: {result}")
                return result
        except Exception as e:
            # Optionally log the error here
            print(f"Error executing Python REPL: {str(e)}")
            raise  # Rethrow the exception to be handled by the caller

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> Any:
        """Execute the tool asynchronously, with error handling."""
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            with change_dir(self.data_directory):
                return await run_in_executor(None, self._run, query)
        except Exception as e:
            # Optionally log the error here
            print(f"Error executing Python REPL asynchronously: {str(e)}")
            raise  # Rethrow the exception to be handled by the caller

# Example testing code within the class file
if __name__ == "__main__":
    # Path to the directory where your CSV file is located
    data_directory = "/home/cvncw/windows/genAI/aiida/aiida_gitlab/data/"

    # Initialize the Python REPL tool with the specified data directory
    repl_tool = PythonREPLPlotTool(data_directory=data_directory)

    # Python code to execute
    python_code = """import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('enriched_mes.csv')
# Get the number of rows
num_rows = len(df)
num_rows"""

    # Run the Python code through the REPL tool and print the result
    output = repl_tool.run(python_code)
    print("Output from Python execution:", repl_tool.run(python_code))