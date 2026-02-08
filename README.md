# Local Project Assistant Agent (LlamaIndex)

This project is a fully local AI agent built using LlamaIndex and an open-weight LLM.
The agent can read a folder of project files (PDF, text, code) and answer questions
about the project. It also uses tools to inspect the project structure.

## Features

- Fully local inference (no API calls)
- Open-weight instruction model (Qwen2.5-3B Instruct, GGUF)
- Retrieval-augmented generation (RAG)
- Agent with tool calling
- Custom tool for scanning project folders

## Project structure

llamaindex-portfolio-agent/
├─ data/
├─ models/
├─ storage/
├─ ingest.py
├─ app.py
├─ tools.py


## Setup

Create a virtual environment and install dependencies:

pip install -r requirements.txt


Download the model:

Qwen2.5-3B-Instruct-Q4_K_M.gguf  
and place it in:

models/


## Ingest documents

Put your files inside:

data/


Then run:

python ingest.py


## Run the agent

python app.py


## Example questions

- What is this project about?
- Which files seem related to model training?
- Scan the project folder and list all files.