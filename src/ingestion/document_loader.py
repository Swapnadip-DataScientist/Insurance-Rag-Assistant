import pymupdf
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}

def discover_documents(raw_data_path: Path):
    """
    Discover supported documents
    """
    documents = []

    for file_path in raw_data_path.rglob("*"):
        if( file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS):
            documents.append(file_path)

    return documents


    
def load_pdf(pdf_path: Path):
    """ 
    Load pdf and extract text page by page
    """

    document = pymupdf.open(pdf_path)

    pages =[]

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append(
            {
                "page_number":page_number,
                "text":text,
                "Character_Count":len(text),
                "is_empty":len(text.strip())==0
            }

        )
    document.close()

    return pages

def load_document(file_path : Path):

    """
    Route a document to the approapiate document type loader
    """
    file_extension = file_path.suffix.lower()

    if file_extension == ".pdf":
        return load_pdf(file_path)

    elif file_extension == ".docx":
        raise NotImplementedError("DOCX loader is not yet built")
    
    elif file_extension == ".txt":
        raise NotImplementedError("DOCX loader is not yet built")

    else:
        raise ValueError(f"Unsupported document type{file_extension}")


if __name__ == "__main__":
    #pdf_path = Path("data/raw/Aviva_Motor_Insurance_policy_default_v15.pdf")
    
    pdf_path = Path("data/raw")

    documents = discover_documents(pdf_path)

    print(f"Total no of documents in the raw folder {len(documents)}")

    for document in documents:
        print(document)
    
            
