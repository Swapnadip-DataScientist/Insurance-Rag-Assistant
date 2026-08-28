from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

def generate_document_id(file_path:str) -> str:
    """
    Generate a stable document ID from the file path.
    The same file path will generate the same document ID.
    """

    return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]
"""
file_path.encode("utf-8") converts the string to bytes. 
hashlib.sha256(...) creates a SHA-256 hash object.
.hexdigest() returns the hash as a hexadecimal string.
"""

def generate_metadata(document : dict) -> dict :

    """
    Generate document level metadata for successfully loaded document. Document structure is 
    file_path, file_type, pages, characters
    """
    file_path = Path(document["file_path"])
    pages = document["pages"]
    total_pages = len(pages)

    total_characters = sum(len(page.get("text","")) for page in pages)

    empty_pages = sum( 1 for page in pages if page.get("is_empty",False))

    metadata ={"document_id" : generate_document_id(str(file_path)),
                "file_name" : file_path.name,
                "file_path" : str(file_path),
                "file_type" : file_path.suffix.lower(),
                "total_page": total_pages,
                "total_characters" :total_characters,
                "empty_pages":empty_pages,
                "ingestion_status":"SUCCESS",
                "ingestion_at":datetime.now(timezone.utc).isoformat()
               }
    return metadata

def save_metadata( metadata : dict, metadata_directory : Path) :

    metadata_directory.mkdir(parents = True, exist_ok = True)

    output_file = (metadata_directory/f"{metadata["document_id"]}.json")

    with open(output_file, "w",encoding = "utf-8") as file:

        json.dump(metadata, file, indent=4, ensure_ascii = False)

        """
        json.dump() -> Writes the Python dictionary into the file as JSON.
        indent=4 -> preety printing 
        ensure_ascii=False -> Allows non-English characters to be stored normally.
        
        """

    return output_file


def generate_batch_metadata( documents : list[dict], metadata_directory : Path):
    """
    Generate and save metadata for all successfully
    loaded documents.
    """

    metadata_records =[]

    for document in documents:

        metadata = generate_metadata(document)

        output_file = save_metadata(metadata,metadata_directory)

        metadata_records.append(metadata)

        print(f"Metadata Created"
              f"{document['file_name']} "
              f"-> {output_file}"
              )

    return metadata_records


if __name__=="__main__":

    from batch_loader import batch_load_documents

    raw_data_path = Path("data/raw")

    metadata_directory = Path("data/metadata")

    document, validation_results = (batch_load_documents(raw_data_path))

    metadata_records = (generate_batch_metadata(document, metadata_directory))

    print("\n****Metadata Summnary***")

    print( f"Metadata Records Created:"
           f"{len(metadata_records)}"
          )
