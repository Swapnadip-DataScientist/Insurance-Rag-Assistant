from pathlib import Path
from batch_loader import batch_load_documents
from text_cleaner import clean_text
from chunker import chunk_pages
from save_chunk import save_chunks_json, save_chunks_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""
project/src/pipeline/chunk_document.py 
-->.parents[0] project/src/pipeline 
-->.parents[1] project/src
-->.parents[2] project
"""

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" 

#*** Batch loader handles discovery + document loading ***#

valid_documents, failed_documents =batch_load_documents(RAW_DATA_PATH)

#*** Processed each load document ***#

for document in valid_documents:


    file_path = Path(document["file_path"])
    pages = document["pages"]

    #print("file_path", document["file_path"])
    #cprint("Pages", document["pages"])

    #***** Clean Page Text for the pages *****

    for page in pages:

        page["text"] = clean_text(page.get("text",""))

    #***** Chunk

    #chunk_pages(pages : List[Dict [str, Any]], document_id : str, source_file : str, max_chunk_chars : int = 500,
     #           overlap_chars : int = 200) -> List [Dict[str, Any]]:

    chunks = chunk_pages(pages =pages,
                        document_id = file_path.stem,
                        source_file = file_path.name, 
                        max_chunk_chars = 500,
                        overlap_chars = 200
                        )

    if not chunks:
        print(f"No chunks available for {file_path.name}")

    #Save Chunks

    save_chunks_json (chunks, PROCESSED_DATA_PATH/F"{file_path.stem}_chunks.json")

    save_chunks_jsonl(chunks, PROCESSED_DATA_PATH/F"{file_path.stem}_chunks.jsonl")

    print(f"Commpleted: {file_path.name} |" f"Chunks: {len(chunks)}" )


    print("Chunking pipeline completed")





