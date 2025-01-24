import gradio as gr
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import openai
from dotenv import load_dotenv
import os
import glob

class DocumentProcessor:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pdf_into_chunks(self, pdf_path, chunk_size=1500, chunk_overlap=150):  
        """  
        Splits the text of an entire PDF into chunks of 'chunk_size' characters,  
        overlapping by 'chunk_overlap' characters. Each chunk identifies the  
        start_page and end_page to indicate the page in which the chunk begins  
        and ends, respectively.  
        """  
        chunks = []  
        try:  
            # Open PDF document with PyMuPDF  
            pdf_document = fitz.open(pdf_path)  
            
            # Gather text from each page and record the text offset at the start of each page  
            text_per_page = []  
            for page_idx, page in enumerate(pdf_document):  
                # Extract plain text for the current page  
                text = page.get_text()  
                # Store (page_number, text) pairs  
                text_per_page.append((page_idx + 1, text))  
    
            # Build a single large text string, recording offsets for page boundaries  
            full_text = ""  
            page_mapping = []  
            for page_number, text in text_per_page:  
                # Record the current length of full_text as the page start offset  
                page_mapping.append((len(full_text), page_number))  
                full_text += text  
    
            # Split full_text into overlapping chunks  
            start_idx = 0  
            text_length = len(full_text)  
            while start_idx < text_length:  
                end_idx = min(start_idx + chunk_size, text_length)  
                chunk_text = full_text[start_idx:end_idx]  
    
                # Determine which pages the start_idx and end_idx-1 belong to  
                start_page = None  
                end_page = None  
    
                for offset, page_num in page_mapping:  
                    if offset <= start_idx:  
                        start_page = page_num  
                    if offset <= end_idx - 1:  
                        end_page = page_num  
    
                # Fallback, in case offsets do not match one of the boundaries  
                if start_page is None:  
                    start_page = page_mapping[0][1]  
                if end_page is None:  
                    end_page = page_mapping[-1][1]  
    
                # Save the chunk along with page-range metadata  
                chunks.append({  
                    "text": chunk_text,  
                    "start_page": start_page,  
                    "end_page": end_page,  
                    "document": pdf_path  
                })  
    
                # Advance the start index by chunk_size minus the overlap  
                start_idx += (chunk_size - chunk_overlap)  
    
        except Exception as e:  
            print(f"Error reading or processing PDF: {e}")  
        return chunks  

    def split_text_into_chunks_langchain(self, pdf_path):
        pdf_document = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in pdf_document)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        return [{"text": chunk, "start_page": 0, "end_page": 0, "document": pdf_path} for chunk in chunks]

class VectorStore:
    def __init__(self, client, collection_name, embedding_model):
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def initialize_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )

    def store_documents(self, chunks):
        texts = [chunk["text"] for chunk in chunks]
        embeddings = list(self.embedding_model.embed(texts))
        
        points = [
            models.PointStruct(
                id=i,
                vector=embeddings[i],
                payload={
                    "text": chunk["text"],
                    "chunknr": i,
                    "start_page": chunk["start_page"],
                    "end_page": chunk["end_page"],
                    "document": chunk["document"]
                }
            ) for i, chunk in enumerate(chunks)
        ]
        
        self.client.upsert(collection_name=self.collection_name, points=points)

class ChatBot:
    def __init__(self, vector_store, embedding_model, language_model, api_url, api_key, use_sources=False, source_tag="Sources:"):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.use_sources = use_sources
        self.source_tag = source_tag
        self.language_model = language_model
        self.api_url = api_url
        self.api_key = api_key
        client = openai.OpenAI(
            api_key=api_key,
            base_url=self.api_url 
        )
        self.openai_client = client

    def _rephrase_query(self, user_input, history):
        if not history:
            return user_input

        history_text = ""
        messages = []
        for msg in history:
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}"
            elif msg["role"] == "assistant":
                assistant_text = msg["content"].split(self.source_tag)[0] if self.source_tag in msg["content"] else msg["content"]
                history_text += f"Assistant: {assistant_text}"
            messages.append({"role": msg["role"], "content": msg["content"]})

        prompt = f"You are an expert at rephrasing text. Your task is to rewrite a given user input based on a given conversation between an assistant and an user. Here is the history of the conversation: {history_text}. The next input of the user is: {user_input}. Please carefully rewrite the last user input considering the given historical chat context. The rewritten user input should contain all necessary information from the previous history. If you are not sure how to rewrite the user input, just return the original user input. Only return the rewritten or original user input."
        response = self.openai_client.chat.completions.create(
            model=self.language_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

    def chat(self, user_input, history):
        if user_input.lower() == "exit":
            exit()

        rephrased_query = self._rephrase_query(user_input, history)
        results = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query=list(self.embedding_model.query_embed(rephrased_query))[0],
            limit=5,
            with_payload=True
        )

        context = "\n".join([point.payload.get("text", "") for point in results.points])
        messages = [msg for msg in history] + [{"role": "user", "content": f"You are a helpful assistant for question-answering tasks. Here is some context for the question: {context} Please carefully consider the given context. Review the user's question and then provide an answer that directly addresses the question using the provided information. If you do not find the answer in the context, say so honestly. User's question: {user_input}"}]
        
        response = self.openai_client.chat.completions.create(
            model=self.language_model,
            messages=messages,
            temperature=0.7
        )
        answer = response.choices[0].message.content

        if self.use_sources:
            sources = "\n".join([f"{i+1}. {point.payload.get('document', '')}, S. {point.payload.get('start_page', '')}"
                               for i, point in enumerate(results.points)])
            return f"{answer}\n\n{self.source_tag}\n{sources}"
        return answer

class ChatbotApp:
    def __init__(self):
        load_dotenv()
        
        # Initialize components
        qdrant_host = os.getenv("QDRANT_HOST")
        self.qdrant_client = QdrantClient(":memory:") if qdrant_host == ":memory:" else QdrantClient(host=qdrant_host)
        
        self.embedding_model = LateInteractionTextEmbedding(os.getenv("EMBEDDINGMODEL_NAME"))
        self.vector_store = VectorStore(self.qdrant_client, os.getenv("QDRANT_COLLECTION_NAME"), self.embedding_model)
        
        self.doc_processor = DocumentProcessor(
            chunk_size=int(os.getenv("CHUNK_SIZE")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP"))
        )
        
        self.chatbot = ChatBot(
            self.vector_store,
            self.embedding_model,
            os.getenv("LLM_MODEL"),
            os.getenv("LLM_URL"),
            os.getenv("LLM_API_KEY"),
            use_sources=os.getenv("USE_SOURCES").lower() == "true",
            source_tag=os.getenv("SOURCES_INDICATOR")
        )

    def prepare_documents(self):
        self.vector_store.initialize_collection()
        subfolder = "docs"
        file_names = glob.glob(os.path.join(subfolder, "**", "*.pdf"), recursive=True)
        
        all_chunks = []
        for file in file_names:
            chunks = (self.doc_processor.split_text_into_chunks_langchain(file) 
                     if not self.chatbot.use_sources 
                     else self.doc_processor.split_pdf_into_chunks(file))
            print(f"Chunks read from file {file}: {len(chunks)}")
            all_chunks.extend(chunks)
        
        self.vector_store.store_documents(all_chunks)

    def launch(self):
        iface = gr.ChatInterface(
            fn=self.chatbot.chat,
            title="Chat with RAG",
            description="Enter your query to chat with RAG sources.",
            type="messages"
        )
        self.prepare_documents()
        iface.launch()

if __name__ == "__main__":
    app = ChatbotApp()
    app.launch()
