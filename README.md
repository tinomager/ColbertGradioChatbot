# Colbert Gradio Chatbot

This repository contains a Gradio-based chatbot that uses the ColBERT model for text embedding and Qdrant for vector storage. The chatbot can process PDF documents, split them into chunks, and use them for retrieval-augmented generation (RAG) to answer user queries.

## Features

- **PDF Processing**: Split PDF documents into chunks for efficient processing.
- **Text Embedding**: Use the ColBERT model for generating text embeddings.
- **Vector Storage**: Store and retrieve embeddings using Qdrant.
- **Gradio Interface**: Interactive chat interface built with Gradio.

## Requirements

- Python 3.8+
- Install dependencies using `pip`:
- Run Ollama with Llama 3.2 3B model locally

```sh
pip install -r requirements.txt
```

## Configuration
The project supports a configuration through python-dotenv.
For this you need to create a .env file in your root project folder with the specific configuration values for your setting.
Configuration includes:
- **QDRANT_HOST** can be set to ":memory" for in process or to the hostname of your Qdrant host, assuming it uses the default port
- **QDRANT_COLLECTION_NAME** defines the name of the collection used or created within Qdrant
- **EMBEDDINGMODEL_NAME** defines the embedding model used by the FastEmbed library
- **USE_SOURCES** defines the algorithm used for chunking. If set to true, a custom algorithm is used that preserves the source pages for the document chunks and if set to false, Langchains text splitter is used
- **SOURCES_INDICATOR** defines a string tag, that indicates the soures in the assistant answer. This should be set to something that would appear in your source documents
- **CHUNK_SIZE** sets the chunk size for chunking
- **CHUNK_OVERLAP** sets the chunk overlap for chunking

## Usage
- **Prepare Documents:** Ensure your PDF documents are ready for processing and place them into the 'docs' folder.
- **Create a .env file** Ensure you created an .env file accordingly to the .sample-env with your specific configuration values
- **Run the Chatbot**: Execute the script to start the Gradio interface.

```sh
python colbert-gradio-chatbot.py
```

## Functions

### split_pdf_into_chunks
Splits a PDF document into chunks.

**Args**:

- pdf_path (str): Path to the PDF document.
- chunk_size (int): Number of characters per chunk.
- overlap_size (int): Number of characters that chunks should overlap.

**Returns**:

- list: A list of chunks with page information.

### split_text_into_chunks_langchain
Splits text into chunks using LangChain.

**Args**:

- pdf_path (str): Path to the PDF document.
- chunk_size (int): Number of characters per chunk.
- overlap_size (int): Number of characters that chunks should overlap.

**Returns**:

- list: A list of chunks with page information.

### prepare_documents
Prepares documents for processing.

### chat_with_rag
Handles user input and generates responses using RAG.

**Args**:

- user_input (str): User's query.
- history (list): Chat history.

**Returns**:

- str: Response to the user's query.

## Open / Known Issues
- currently the used docs are hardcoded and need to be read from the 'docs' directory, this was done for testing purposes
- source citication generation has to be validated and test

## License
This project is licensed under the MIT License.