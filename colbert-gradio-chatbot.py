import gradio as gr
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import ollama

qdrant_client = QdrantClient(":memory:")
qdrant_collection_name = "colbert_collection"
model_name = "answerdotai/answerai-colbert-small-v1"
embedding_model = LateInteractionTextEmbedding(model_name)
use_sources = True
source_tag = "Sources -->"

def split_pdf_into_chunks(pdf_path, chunk_size, overlap_size):
    """
    Teilt ein PDF-Dokument in Chunks auf.
    
    Args:
        pdf_path (str): Der Pfad zum PDF-Dokument.
        chunk_size (int): Die Anzahl der Zeichen pro Chunk.
        overlap_size (int): Die Anzahl der Zeichen, die sich die Chunks überlappen sollen.
        
    Returns:
        list: Eine Liste von Chunks mit den zugehörigen Seiteninformationen im Format {"text": str, "start_page": int, "end_page": int}.
    """
    chunks = []
    try:
        # PDF-Dokument öffnen

        pdf_document = fitz.open(pdf_path)
        
        # Den gesamten Text extrahieren
        text_per_page = []
        page_idx = 0
        for page in pdf_document:
            text = page.get_text()
            text_per_page.append((page_idx + 1, text))  # Merkt sich die Seitenzahl
            page_idx += 1
            
        # Alle Texte zu einer langen Zeichenkette zusammenfügen
        full_text = ""
        page_mapping = []  # Merkt sich, wo die Chunks im Text anfangen
        for page_number, text in text_per_page:
            page_mapping.append((len(full_text), page_number))  # Startindex -> Seitenzahl
            full_text += text
        
        # Chunks erstellen
        start_idx = 0
        while start_idx < len(full_text):
            end_idx = min(start_idx + chunk_size, len(full_text))
            chunk_text = full_text[start_idx:end_idx]
            
            # Seiten des Chunks bestimmen
            start_page = None
            end_page = None
            
            for idx, page in page_mapping:
                if idx <= start_idx:
                    start_page = page
                if idx >= end_idx - 1 and end_page is None:
                    end_page = page
                    break
            
            if end_page is None:
                end_page = page_mapping[-1][1]
            
            chunks.append({
                "text": chunk_text,
                "start_page": start_page,
                "end_page": end_page, 
                "document": pdf_path
            })
            
            # Startindex für den nächsten Chunk
            start_idx += chunk_size - overlap_size

    except Exception as e:
        print(f"Fehler: {e}")

    return chunks

def split_text_into_chunks_langchain(pdf_path, chunk_size, overlap_size):
    pdf_document = fitz.open(pdf_path)

    # Text extrahieren
    text = ""
    for page in pdf_document:
        text += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size
    )

    chunks = text_splitter.split_text(text)
    chunk_objects = []
    for i, chunk in enumerate(chunks):
        chunk_objects.append({
            "text": chunk,
            "start_page": 0,
            "end_page": 0,
            "document": pdf_path
        })

    return chunk_objects

def prepare_documents():
    file_names = ["docs/t-phone-pro-betriebsanleitung.pdf", "docs/elterngeld-elterngeldplus-und-elternzeit-data.pdf", "docs/ma1622_iphone_ios5_user_guide.pdf"]
    all_chunks = []

    for file in file_names:
        if not use_sources:
            chunks = split_text_into_chunks_langchain(file, 1000, 150)
        else:  
            chunks = split_pdf_into_chunks(file, 1000, 150)

        print(f"Chunks read from file {file}: {len(chunks)}")
        all_chunks.extend(chunks)

    if not qdrant_client.collection_exists(qdrant_collection_name):
        qdrant_client.create_collection(
            collection_name=qdrant_collection_name,
            vectors_config=models.VectorParams(
                size=128, #size of each vector produced by ColBERT
                distance=models.Distance.COSINE, #similarity metric between each vector
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM #similarity metric between multivectors (matrices)
                ),
            ),
        )
        print("Collection created")

    all_texts = [chunk["text"] for chunk in all_chunks]
    text_embeddings = list(
        embedding_model.embed(all_texts)
    )
    points = []
    for i, chunk in enumerate(all_chunks):
        points.append(models.PointStruct(
            id=i,
            vector=text_embeddings[i],
            payload={
                "text": chunk["text"], 
                "chunknr":i,
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "document": chunk["document"]
            }
        ))
    
    qdrant_client.upsert(
        collection_name=qdrant_collection_name,
        points=points
    )

# Define your chat function
def chat_with_rag(user_input, history):
    if user_input.lower() == "exit":
        exit()

    messages = []
    
    rephrase_response = user_input
    if len(history) > 0:
        history_text = ""
        for msg in history:
            if msg["role"] == "user":
                history_text += f"User: {msg["content"]}"
            elif msg["role"] == "assistant":
                assistant_text = msg["content"]
                if source_tag in assistant_text:
                    #Remove the sources from the assistant text so the model isn't confused
                    assistant_text = assistant_text.split(source_tag)[0]
                history_text += f"Assistant: {assistant_text}"
            messages.append({"role" : msg["role"], "content" : msg["content"]})
        prompt = f"You are an expert at rephrasing text. Your task is to rewrite a given user input based on a given conversation between an assistant and an user. Here is the history of the conversation: {history_text}. The next input of the user is: {user_input}. Please carefully rewrite the last user input considering the given historical chat context. The rewritten user input should contain all necessary information from the previous history. If you are not sure how to rewrite the user input, just return the original user input. Only return the rewritten or original user input."
        rephrase_response = ollama.generate(model="llama3.2", prompt=prompt).response
     
    res = qdrant_client.query_points(
        collection_name=qdrant_collection_name,
        query=list(embedding_model.query_embed(rephrase_response))[0], #converting generator object into numpy.ndarray
        limit=5, #How many closest to the query movies we would like to get
        with_payload=True #So metadata is provided in the output
    )

    context = "\n".join([point.payload.get("text", "") for point in res.points])
    prompt = f"You are a helpful assistant for question-answering tasks. Here is some context for the question: {context} Please carefully consider the given context. Review the user's question and then provide an answer that directly addresses the question using the provided information. If you do not find the answer in the context, say so honestly. User's question: {user_input}"
    messages.append({"role" : "user", "content" : prompt})
    response = ollama.chat(model="llama3.2", messages=messages)
    
    answer = response.message.content
    if use_sources:
        idx_number = 1
        sources = ""
        for point in res.points:
            sources += f"{idx_number}. {point.payload.get('document', '')}, S. {point.payload.get("start_page", "")}\n\n"
            idx_number += 1
        return answer + f"\n\n{source_tag}\n" + sources
    else:
        return answer

    

# Create the Gradio interface
iface = gr.ChatInterface(
    fn=chat_with_rag,
    title="Chat with RAG",
    description="Enter your query to chat with RAG sources.",
    type="messages"
)

prepare_documents()
# Launch the interface
iface.launch()