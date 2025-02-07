# main.py
import nest_asyncio
nest_asyncio.apply()  # Allow asyncio to run in Colab's notebook environment

import gradio as gr
import asyncio
from model import Inference  # Your asynchronous Inference class

# Instantiate our Inference object
inference = Inference()

def add_document_sync(user_uuid, pdf_file):
    """
    Synchronous wrapper for adding a document.
    The pdf_file input from Gradio returns the local file path.
    """
    if not user_uuid:
        return "Error: User UUID is required."
    if not pdf_file:
        return "Error: Please upload a PDF file."

    pdf_path = pdf_file  # Gradio's File component returns the file path
    try:
        # Run the asynchronous method
        result = asyncio.run(inference.add_document(user_uuid, pdf_path))
        return result
    except Exception as e:
        return f"Error during add_document: {str(e)}"

def infer_sync(user_uuid, question):
    """
    Synchronous wrapper for performing inference.
    """
    if not user_uuid:
        return "Error: User UUID is required."
    if not question:
        return "Error: Please enter a question."
    try:
        # Run the asynchronous method
        answer = asyncio.run(inference.infer(user_uuid, question))
        return answer
    except Exception as e:
        return f"Error during inference: {str(e)}"

# Build the Gradio interface with two tabs
with gr.Blocks() as demo:
    gr.Markdown("# Document Search & Inference UI")
    
    with gr.Tabs():
        # Tab for adding a document
        with gr.TabItem("Add Document"):
            gr.Markdown("Upload a PDF file along with your user UUID to add the document.")
            with gr.Row():
                user_uuid_input = gr.Textbox(label="User UUID", placeholder="Enter your UUID here")
                pdf_input = gr.File(label="Upload PDF", file_types=['.pdf'])
            add_button = gr.Button("Add Document")
            add_status = gr.Textbox(label="Status", interactive=False)
            add_button.click(add_document_sync, inputs=[user_uuid_input, pdf_input], outputs=add_status)
        
        # Tab for asking a question (inference)
        with gr.TabItem("Ask Question"):
            gr.Markdown("Enter your user UUID and question to get an answer based on your document content.")
            with gr.Row():
                user_uuid_input_q = gr.Textbox(label="User UUID", placeholder="Enter your UUID here")
                question_input = gr.Textbox(label="Question", placeholder="Enter your question here")
            infer_button = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer", interactive=False)
            infer_button.click(infer_sync, inputs=[user_uuid_input_q, question_input], outputs=answer_output)

# Launch the Gradio UI with share=True so that Colab gives you a public URL.
demo.launch(share=True)
