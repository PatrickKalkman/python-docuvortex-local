# Harnessing Local Language Models - A Guide to Transitioning From OpenAI to On-Premise Power

## Exploring alternative large language models for enhanced document management

![DocuVortex](/article_image.jpg "DocuVortex")

This project aims to implement a document-based question-answering system using a local LLM model, Python, and the Langchain Framework. It processes PDF documents, breaking them into ingestible chunks, and then stores these chunks into a Chroma DB vector database for querying. It complements a Medium article called [Harnessing Local Language Models - A Guide to Transitioning From OpenAI to On-Premise Power](https://medium.com/@pkalkman).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

To install the project, you need to have [Python](https://www.python.org/downloads/) installed on your machine. Also you need a machine with a cuda compatible GPU. If you want to run the application on the CPU, you need to change the cuda references in the code to cpu.

### Installing

The project uses several dependencies. After cloning the repository, navigate to the project directory and install dependencies with the following commands:

```bash
pip install -r requirements
```

## Running the Application


### Ingesting Documents
To ingest documents, place your PDF files in the 'docs' folder make sure that you are in the app folder and run the following command:

```bash
cd app
python ingest.py
```

### Querying Documents
To query the ingested documents, make sure that you are in the app folder, run the following command and follow the interactive prompts:

```bash
cd app
python query.py
```


### Authors
[Patrick Kalkman](https://github.com/PatrickKalkman)

### License
This project is licensed under the MIT license - see the LICENSE.md file for details

### Acknowledgments
- [Langchain Framework](https://python.langchain.com/en/latest/index.html)
- [Chroma DB](https://www.trychroma.com/)
