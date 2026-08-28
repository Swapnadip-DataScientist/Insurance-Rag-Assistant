from pathlib import Path

from document_loader import discover_documents, load_document

def validate_document(file_path : Path, pages : list[dict]):
    """
    Validate the document after text extraction

    Checks
    - document contains extracted pages
    - extracted text is empty or not
    - counts of empty pages
    - calculayes total extracted characters
    """

    total_pages = len(pages)

    total_characters = sum(len(page.get("text","")) for page in pages)

    empty_pages     = sum(1 for page in pages if not page.get("text","").strip())

    if total_pages ==0:
        status = "FAILED"

    elif total_characters ==0:
        status ="FAILED"

    elif empty_pages==total_pages:
        status ="FAILED"

    elif empty_pages>0:
        status ="WARNING"

    else:
        status ="SUCCESS"

    return {
        "file_name":file_path.name,
        "file_path":str(file_path),
        "total_pages":total_pages,
        "total_characters":total_characters,
        "empty_pages":empty_pages,
        "status":status
    }



def batch_load_documents(raw_data_path: Path):

    """
    Discover, load and validate all supported documents. 
    
    A failure in one document should not stop the entire batch.
    """

    file_paths = discover_documents(raw_data_path)

    loaded_document = []
    validation_results = []

    for file_path in file_paths:

        try:

            pages = load_document(file_path)

            validation = validate_document(file_path =file_path, pages =pages)

            validation_results.append(validation)

            if validation["status"] !="FAILED":

                    loaded_document.append(
                        {
                            "file_path" :str(file_path),
                            "file_name" :file_path.name,
                            "pages":pages

                        }

                    )

        except Exception as error:

            validation_results.append(
                {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "total_pages": 0,
                    "total_characters" :0,
                    "empty_pages":0,
                    "status": "FAILED",
                    "error" : str(error)

                }

            )

            for results in validation_results:

                if results["status"]=="FAILED":
                    print(
                        f"{results["file_name"]} |"
                        f"Pages{results["total_pages"]} | "
                        f"Characters{results["total_characters"]} | "
                        f"Empty Pages{results["empty_pages"]} | "
                    )



    return loaded_document, validation_results


if __name__ =="__main__":

    raw_data_path = Path("data/raw")

    documents, validation_results = batch_load_documents(raw_data_path)

    print("\n===Document Batch Load Summary")
    print("*" *20)

    print(f"Document Discovered: {len(validation_results)}")
    print(f"Valid Documents: {len(documents)}")

    for result in validation_results:
        print(result)