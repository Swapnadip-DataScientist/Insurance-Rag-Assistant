import json
from pathlib import Path
from typing import List, Dict, Any

def save_chunks_json(chunks : List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Save chunks as a JSON file.
    Parameters
    ----------
    chunks:   List of chunk dictionaries.
    output_path: Full destination path.
    Example:  data/processed/aviva_motor_chunks.json
    Returns
    -------
    Path   Saved file path.
    """
    if not chunks:
        raise ValueError("No chunks provided to save")

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok = True)

    with output_path.open(mode="w", encoding ="utf-8") as file:

        json.dump(chunks, file, ensure_ascii=False, indent = 2)

    return output_path

def save_chunks_jsonl(chunks: List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Save chunks using JSON Lines format. JSONL stores one chunk per line.
    This format is preferable later for larger ingestion pipelines because each record can be processed independently.
    """

    if not chunks:
        raise ValueError("No Chunks provided to save")

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok = True)

    with output_path.open(mode="w", encoding="utf-8") as file:

        for chunk in chunks:

            json_line = json.dumps(chunk, ensure_ascii=False)

            file.write(json_line + "\n")

    return output_path


def load_chunks_json(input_path : Path) -> List[Dict[str, Any]]:

    """
    Load previously saved chunk JSON file.
    """
    input_path = Path(input_path)

    if not input_path.exists():

        return FileNotFoundError(f"Chunk file is not available in the mentioned {input_path}")

    with input_path.open(
        mode="r", 
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return chunks


def load_chunks_jsonl(input_path : Path) -> List[Dict[str, Any]]:

    """
    Load chunks saved JSON file.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        return FileNotFoundError(f"Chunk file is not available in the mentioned {input_path}")

    chunks =[]

    with input_path.open(
        mode="r", 
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)

                chunks.append(chunk)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line"
                    f"{line_number} of {input_path}"
                ) from exc

    return chunks