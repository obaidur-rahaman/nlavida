# Description: This script is used to load the prompt from the file and augment it with user provided description of the file if it exists

from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate

import pickle
import os

def get_prompt(root_directory: str) -> str:
    # Path where the serialized prompt is stored
    prompt_path = root_directory + "/prompt/core_prompt.pkl"

    # Load the prompt from the file
    with open(prompt_path, 'rb') as f:
        prompt = pickle.load(f)

    # Augment the prompt with some system instruction
    with open(root_directory + "/prompt/system_prompt.txt", 'r') as file:
            additional_prompt = file.read()

    # Augment the prompt with user provided description of the file if it exists
    # First check if the file exists
    if os.path.exists(root_directory + "/prompt/user_description_of_file.txt"):  
        print("User description file exists\n")
        with open(root_directory + "/prompt/user_description_of_file.txt", 'r') as file:
            user_description = file.read()
            # Now join the system prompt and user description
            additional_prompt = additional_prompt + "\n" + user_description
    
    # Create a PromptTemplate instance
    prompt_template = PromptTemplate(
        template=additional_prompt,
        input_variables=["input"]
    )

    # Create a SystemMessagePromptTemplate instance with the PromptTemplate
    system_message_prompt = SystemMessagePromptTemplate(
        template=additional_prompt,
        prompt=prompt_template
    )

    prompt.messages[0] = system_message_prompt

    #print(prompt.format(input="", chat_history=[], agent_scratchpad=[]))
    return(prompt)

if __name__ == "__main__":
    root_directory = "/Users/obaidurrahaman/Documents/aiidav/aiidav_github/"
    prompt = get_prompt(root_directory)