from __future__ import annotations

import os

import gradio as gr
import httpx


# ============================================================
# CONFIGURATION
# ============================================================

FASTAPI_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000/ask",
)

# Local CPU inference can take more than a minute,
# so keep a generous timeout.
API_TIMEOUT_SECONDS = float(
    os.getenv(
        "RAG_API_TIMEOUT_SECONDS",
        "240",
    )
)


# ============================================================
# CALL FASTAPI
# ============================================================

def ask_rag(query: str):
    """
    Send the user question to the FastAPI /ask endpoint.

    Gradio is only the presentation layer.
    The actual RAG pipeline remains behind FastAPI.
    """

    query = query.strip()

    if not query:
        return (
            "validation_error",
            "Please enter an insurance policy question.",
            "",
            "None",
            "None",
            [],
            0.0,
        )

    try:
        with httpx.Client(
            timeout=API_TIMEOUT_SECONDS
        ) as client:

            response = client.post(
                FASTAPI_URL,
                json={
                    "query": query
                },
            )

            response.raise_for_status()

        data = response.json()

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    except httpx.TimeoutException:

        return (
            "timeout",
            "The RAG service took too long to respond.",
            "",
            "None",
            "None",
            [],
            0.0,
        )

    # --------------------------------------------------------
    # FastAPI returned 4xx / 5xx
    # --------------------------------------------------------

    except httpx.HTTPStatusError as exc:

        try:
            error_data = exc.response.json()

            detail = error_data.get(
                "detail",
                "FastAPI request failed.",
            )

        except Exception:
            detail = "FastAPI request failed."

        return (
            "api_error",
            str(detail),
            "",
            "None",
            "None",
            [],
            0.0,
        )

    # --------------------------------------------------------
    # FastAPI unavailable
    # --------------------------------------------------------

    except httpx.RequestError:

        return (
            "service_unavailable",
            (
                "Unable to connect to the FastAPI service. "
                "Check that FastAPI is running on port 8000."
            ),
            "",
            "None",
            "None",
            [],
            0.0,
        )

    # ========================================================
    # FORMAT SUCCESSFUL RESPONSE
    # ========================================================

    answer_status = data.get(
        "answer_status",
        "",
    )

    answer = data.get(
        "answer",
        "",
    )

    citations = ", ".join(
        data.get(
            "citations",
            [],
        )
    )

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    conditions = data.get(
        "conditions",
        [],
    )

    if conditions:
        conditions_text = "\n".join(
            f"• {item}"
            for item in conditions
        )
    else:
        conditions_text = "None"

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    limitations = data.get(
        "limitations",
        [],
    )

    if limitations:
        limitations_text = "\n".join(
            f"• {item}"
            for item in limitations
        )
    else:
        limitations_text = "None"

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    source_rows = []

    for source in data.get(
        "sources",
        [],
    ):
        source_rows.append(
            [
                source.get(
                    "citation_id",
                    "",
                ),
                source.get(
                    "source_file",
                    "",
                ),
                source.get(
                    "page_number",
                    "",
                ),
                source.get(
                    "page_chunk_index",
                    "",
                ),
            ]
        )

    # --------------------------------------------------------
    # Convert latency from milliseconds to seconds
    # --------------------------------------------------------

    latency_seconds = round(
        float(
            data.get(
                "latency_ms",
                0.0,
            )
        )
        / 1000,
        2,
    )

    return (
        answer_status,
        answer,
        citations,
        conditions_text,
        limitations_text,
        source_rows,
        latency_seconds,
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Insurance RAG Assistant"
) as demo:

    gr.Markdown(
        """
# Insurance RAG Assistant

Ask a question about the insurance policy documents
available in the knowledge base.

The assistant retrieves relevant policy evidence,
reranks it and generates a grounded answer with citations.
"""
    )

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    query_input = gr.Textbox(
        label="Insurance Policy Question",
        placeholder=(
            "Example: Will Aviva cover a home charging "
            "point rated above 32 amps?"
        ),
        lines=3,
    )

    ask_button = gr.Button(
        "Ask Policy Question",
        variant="primary",
    )

    # --------------------------------------------------------
    # Status + latency
    # --------------------------------------------------------

    with gr.Row():

        status_output = gr.Textbox(
            label="Answer Status",
            interactive=False,
        )

        latency_output = gr.Number(
            label="Response Time (seconds)",
            interactive=False,
        )

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    answer_output = gr.Textbox(
        label="Answer",
        lines=5,
        interactive=False,
    )

    citations_output = gr.Textbox(
        label="Citations",
        interactive=False,
    )

    # --------------------------------------------------------
    # Conditions / Limitations
    # --------------------------------------------------------

    with gr.Row():

        conditions_output = gr.Textbox(
            label="Conditions",
            lines=5,
            interactive=False,
        )

        limitations_output = gr.Textbox(
            label="Limitations",
            lines=5,
            interactive=False,
        )

    # --------------------------------------------------------
    # Source evidence
    # --------------------------------------------------------

    sources_output = gr.Dataframe(
        headers=[
            "Citation",
            "Source File",
            "Page",
            "Chunk",
        ],
        datatype=[
            "str",
            "str",
            "number",
            "number",
        ],
        interactive=False,
        label="Evidence Sources",
    )

    # ========================================================
    # BUTTON EVENT
    # ========================================================

    ask_button.click(
        fn=ask_rag,
        inputs=[
            query_input,
        ],
        outputs=[
            status_output,
            answer_output,
            citations_output,
            conditions_output,
            limitations_output,
            sources_output,
            latency_output,
        ],
        concurrency_limit=1,
    )


# ============================================================
# QUEUE
# ============================================================

demo.queue(
    default_concurrency_limit=1,
    max_size=10,
)


# ============================================================
# START GRADIO
# ============================================================

"""if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )"""
if __name__ == "__main__":

    demo.launch(
        server_name=os.getenv(
            "GRADIO_SERVER_NAME",
            "127.0.0.1",
        ),
        server_port=int(
            os.getenv(
                "GRADIO_SERVER_PORT",
                "7860",
            )
        ),
        show_error=True,
    )