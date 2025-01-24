import fitz
import os, sys

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

def test_split_pdf_into_chunks():  
    """  
    Test the split_pdf_into_chunks function with a sample PDF.  
    Replace 'sample.pdf' with a path to an actual PDF file on your system.  
    """  
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    pdf_file = os.path.join(parent_dir, "docs", "t-phone-pro-betriebsanleitung.pdf")  # Provide a real PDF file path  
    chunk_size = 1000         # Smaller chunk size to test easily  
    chunk_overlap = 150  
  
    document_processor = DocumentProcessor(1000, 150)
    # Call the function  
    chunks = document_processor.split_pdf_into_chunks(pdf_file)  
  
    # Print out results  
    print(f"Total chunks created: {len(chunks)}")  
    for i, chunk in enumerate(chunks):  
        print(f"Chunk {i+1}:")  
        print(f"  Pages: {chunk['start_page']} -> {chunk['end_page']}")  
        print(f"  Text length: {len(chunk['text'])}")  
        print(f"  Sample text start: {chunk['text'][:50]!r}...")  
        print("-" * 30)  
   
# Example usage:  
if __name__ == "__main__":  
    test_split_pdf_into_chunks()  