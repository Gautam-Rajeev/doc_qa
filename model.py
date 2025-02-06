import re
import gc
import os
import torch
import pandas as pd
import numpy as np
import openai
import google.generativeai as genai
try:
    from marker.converters.pdf import PdfConverter
except ModuleNotFoundError:
    from marker_pdf.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from json_repair import repair_json
import asyncio


def cleanup_memory():
    """
    Cleans up memory by running the garbage collector and clearing GPU cache (if available).
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared.")
    print("Memory cleanup completed.")

cleanup_memory()

class Chunker:
    def __init__(self, config: dict = None):
        if config is None:
            config = {"output_format": "markdown", "ADDITIONAL_KEY": "VALUE"}
        self.config_parser = ConfigParser(config)
        self.artifact_dict = create_model_dict()
        self.converter = PdfConverter(
            config=self.config_parser.generate_config_dict(),
            artifact_dict=self.artifact_dict,
            processor_list=self.config_parser.get_processors(),
            renderer=self.config_parser.get_renderer()
        )

    @staticmethod
    def clean_heading(text: str) -> str:
        text = re.sub(r'[#*]+', '', text)
        return text.strip()

    @staticmethod
    def is_table(block: str) -> bool:
        return bool(re.search(r'\|[\s-]+\|', block))

    def chunk(self, input_pdf: str) -> pd.DataFrame:
        try:
            markdown_output = self.converter(input_pdf)
        except MemoryError:
            print("MemoryError: Not enough memory to process the PDF.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return pd.DataFrame()

        markdown_text = markdown_output.markdown
        toc_entries = markdown_output.metadata.get('table_of_contents', [])
        blocks = re.split(r'\n\s*\n', markdown_text)
        rows = []
        heading_stack = {}
        bbox_stack = {}
        page_stack = {}
        toc_index = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if block.startswith("#"):
                block_type = "heading"
            elif self.is_table(block):
                block_type = "table"
            else:
                block_type = "text"

            if block_type == "heading":
                match = re.match(r'^(#+)', block)
                level = len(match.group(1)) if match else 1
                heading_text = self.clean_heading(block)
                keys_to_remove = [lvl for lvl in heading_stack if lvl >= level]
                for lvl in keys_to_remove:
                    heading_stack.pop(lvl, None)
                    bbox_stack.pop(lvl, None)
                    page_stack.pop(lvl, None)

                heading_stack[level] = heading_text
                if toc_index < len(toc_entries):
                    toc_item = toc_entries[toc_index]
                    bbox_stack[level] = toc_item.get('polygon')
                    page_stack[level] = toc_item.get('page_id')
                    toc_index += 1
                else:
                    bbox_stack[level] = None
                    page_stack[level] = None

                hierarchy_titles = " > ".join([heading_stack[lvl] for lvl in sorted(heading_stack.keys())])
                current_bbox = bbox_stack.get(level, None)
                current_page = page_stack.get(level, None)

                rows.append({
                    "text": block,
                    "titles": hierarchy_titles,
                    "type": "heading",
                    "bbox": current_bbox,
                    "start_page": current_page,
                    "end_page": current_page
                })
            else:
                if heading_stack:
                    hierarchy_titles = " > ".join([heading_stack[lvl] for lvl in sorted(heading_stack.keys())])
                    deepest_level = max(heading_stack.keys())
                    current_bbox = bbox_stack.get(deepest_level, None)
                    current_page = page_stack.get(deepest_level, None)
                else:
                    hierarchy_titles = None
                    current_bbox = None
                    current_page = None

                rows.append({
                    "text": block,
                    "titles": hierarchy_titles,
                    "type": block_type,
                    "bbox": current_bbox,
                    "start_page": current_page,
                    "end_page": current_page
                })

        df = pd.DataFrame(rows)
        final_df = df.loc[df['type'].isin(['text', 'table']), :].copy()
        final_df['content'] = final_df['titles'].astype(str) + '\n' + final_df['text']
        gc.collect()
        return final_df


class MultiUserDocumentSearch:
    def __init__(self, chunker, embedding_model_name='all-MiniLM-L6-v2', authorized_users=None):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunker = chunker
        if authorized_users is None:
            self.authorized_users = {
                'afde9394-bebd-4c5b-b794-7c3a53e5885e',
                '428e5aa6-9f00-4b84-8dc2-5167d4037301',
                '524fa7b3-2786-4658-aba4-1d34fafca25d',
                '97c2441a-8f04-465a-bfd1-822d0a9a35b8'
            }
        else:
            self.authorized_users = set(authorized_users)
        self.df_chunks = pd.DataFrame(columns=['user', 'content', 'text', 'titles'])
        self.corpus_embeddings = None

    def embed_chunks(self, new_chunks):
        new_embeddings = self.embedding_model.encode(new_chunks)
        return new_embeddings

    def add_document(self, user, document_path):
        if user not in self.authorized_users:
            raise ValueError("User not authorized to add documents.")
        chunks_df = self.chunker.chunk(document_path)
        chunks_df['user'] = user
        new_chunks = chunks_df['content'].tolist()
        new_embeddings = self.embed_chunks(new_chunks)
        self.df_chunks = pd.concat([self.df_chunks, chunks_df], ignore_index=True)
        if self.corpus_embeddings is None:
            self.corpus_embeddings = new_embeddings
        else:
            self.corpus_embeddings = np.vstack([self.corpus_embeddings, new_embeddings])
        print(f"Document added for user {user}. Total chunks: {len(self.df_chunks)}")

    def search(self, user, query, k=3, return_columns=['text','titles']):
        if user not in self.authorized_users:
            raise ValueError("User not authorized to search documents.")
        user_df = self.df_chunks[self.df_chunks['user'] == user]
        if user_df.empty:
            print(f"No documents found for user {user}.")
            return pd.DataFrame()
        user_indices = user_df.index.to_list()
        user_embeddings = self.corpus_embeddings[user_indices]
        query_embedding = self.embedding_model.encode([query])[0]
        sims = cosine_similarity([query_embedding], user_embeddings)[0]
        top_k_order = np.argsort(sims)[-k:][::-1]
        results = []
        for i in top_k_order:
            result_dict = {'user': user, 'Similarity': sims[i]}
            for col in return_columns:
                result_dict[col] = user_df.iloc[i][col]
            results.append(result_dict)
        results_df = pd.DataFrame(results)
        return results_df



class LLM_call_processor:
    def __init__(self, llm='openai'):
        self.llm = llm
        self.api_key = None
        self.model = None

    def set_api_key(self, api_key):
        self.api_key = api_key
        if self.llm == 'gemini':
            genai.configure(api_key=self.api_key)
        elif self.llm == 'openai':
            openai.api_key = self.api_key
        else:
            raise ValueError(f"LLM provider {self.llm} not supported")

    def llm_call(self, messages, model=None, type_json=True):
        if not model and self.model:
            model = self.model
        if self.llm == 'openai':
            if not openai.api_key:
                raise ValueError("OpenAI API key not set. Use set_api_key().")
            if not model:
                model = "gpt-4o-mini"
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages
            )
            content = response.choices[0].message.content
        elif self.llm == 'gemini':
            if not self.api_key:
                raise ValueError("Gemini API key not set. Use set_api_key().")
            if not model:
                model = "gemini-1.5-flash"
            genai_model = genai.GenerativeModel(model)
            response = genai_model.generate_content(messages)
            content = response.text
        else:
            raise ValueError(f"LLM provider {self.llm} not supported")
        if type_json:
            return repair_json(content)
        else:
            return content

    def create_text_message(self, system_prompt='', user_prompt='Hi'):
        if self.llm == 'openai':
            text_message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        elif self.llm == 'gemini':
            text_message = str(system_prompt) + ' ' + str(user_prompt)
        return text_message

    def process_text(self, system_prompt, user_prompt, type_json=True):
        messages = self.create_text_message(system_prompt, user_prompt)
        return self.llm_call(messages, type_json=type_json)


class Inference:

    def __init__(self):
        self.chunker = Chunker()
        self.search_system = MultiUserDocumentSearch(chunker=self.chunker)
        self.processor_llm = LLM_call_processor(llm='gemini')
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key:
            self.processor_llm.set_api_key(gemini_api_key)
        else:
            raise ValueError("GEMINI_API_KEY not set in environment")

    async def add_document(self, user, document_path):
        # Run the CPU-bound add_document call in a thread.
        await asyncio.to_thread(self.search_system.add_document, user, document_path)
        return "Document added successfully."

    async def infer(self, user, question):
        # Run the search in a thread.
        results = await asyncio.to_thread(self.search_system.search, user, question, 3, ['text','titles'])
        if results.empty:
            return "No relevant content found."
        content_chunk = results[['text', 'titles']].to_json()
        system_prompt = (
            "You are an AI RAG system that answers questions based on content.\n"
            "If the answer is not found in the content, say you do not know the answer.\n"
            f"The content is: {content_chunk}\n"
            "The user question is:"
        )
        # Run the LLM call in a thread.
        answer = await asyncio.to_thread(self.processor_llm.process_text, system_prompt, question, False)
        return answer
