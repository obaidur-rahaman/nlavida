import pickle
from langchain import hub
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate

# Pull the prompt from the hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Path where you want to store the serialized prompt
prompt_path = "../prompt/core_prompt.pkl"

# Serialize and save the prompt
with open(prompt_path, 'wb') as f:
    pickle.dump(prompt, f)
