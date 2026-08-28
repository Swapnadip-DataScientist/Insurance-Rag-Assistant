import re

def clean_text (text : str) -> str:

    """
    Perform conservative text cleaning for insurance documents.

    Cleaning performed:
    - remove null characters
    - normalize non-breaking spaces
    - normalize Windows/Mac line endings
    - remove excessive spaces and tabs
    - remove spaces before punctuation
    - collapse excessive blank lines

    Important:
    We intentionally do NOT remove numbers, punctuation,
    section references, currency symbols, percentages, etc.
    """

    if not text:

        return ""

    text = text.replace("\x00", "") # Remove null characters
    text = text.replace("\xa0", " ")# Replace non-breaking spaces 

    text = text.replace("\r\n", "\n")# Normalize line endings
    text = text.replace("\r", "\n")

    cleaned_lines = []

    for line in text.split("\n"):

       line = line.strip() #Remove spaces
       line = re.sub(r"[ \t]+", "", line) # Replace multiple spaces/tabs with one space
       line = re.sub(r"\s+([,.;:!?])",r"\1",line)

       cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Maximum two consecutive new lines
    text = re.sub(r"\n{3,}","\n\n",text)

    return text.strip()




def clean_document_pages(pages : list[dict] ) -> list[dict]:

    """
    Clean all pages belonging to one document. Original extracted text is preserved. A new field called 'cleaned_text' is added.
    
    """

    cleaned_pages = []

    for page in pages:

        original_text = page.get("text", "")

        cleaned_text = clean_text(original_text)

        cleaned_page = page.copy()

        cleaned_page["cleaned_text"] = cleaned_text

        cleaned_page["cleaned_character_count"] = len(clean_text)

        cleaned_page["is_cleaned_empty"] = len(cleaned_text.strip()) ==0

        cleaned_pages.append(cleaned_page)

    return cleaned_pages


def clean_documents(documents:list[dict]) -> list[dict]:

    """
    Clean every successfull loaded document. 
    Expected Input {
    "file_path" :"...",
    "file_name" :"...",
    "pages" :[...]
    }
    """
    cleaned_documents = []

    for document in documents:

        cleaned_document = document.copy()

        cleaned_document["pages"] = clean_document_pages(document["pages"])

        cleaned_documents.append(cleaned_document)


    return cleaned_documents

    """{
    "page_number": 5,

    "text": "original PyMuPDF extraction...",

    "cleaned_text": "cleaned version...",

    "cleaned_character_count": 2341,

    "is_cleaned_empty": False
    }   """


    

