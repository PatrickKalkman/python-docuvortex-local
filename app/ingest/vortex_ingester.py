from typing import List
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.vectorstores import Chroma
import langchain.docstore.document as docstore
from loguru import logger

from settings import COLLECTION_NAME, PERSIST_DIRECTORY

from .vortex_pdf_parser import VortexPdfParser
from .vortext_content_iterator import VortexContentIterator


class VortexIngester:

    def __init__(self, content_folder: str):
        self.content_folder = content_folder

    def ingest(self) -> None:
        vortex_content_iterator = VortexContentIterator(self.content_folder)
        vortex_pdf_parser = VortexPdfParser()

        chunks: List[docstore.Document] = []
        for pdf_document in vortex_content_iterator:
            vortex_pdf_parser.set_pdf_file_path(pdf_document)
            document_chunks = vortex_pdf_parser.clean_text_to_docs()
            chunks.extend(document_chunks)
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_document}")

        instructor_embeddings = HuggingFaceInstructEmbeddings(
                                  model_name="hkunlp/instructor-xl",
                                  model_kwargs={"device": "cuda"})

        logger.info("Loaded embeddings")
        vector_store = Chroma.from_documents(
            chunks,
            instructor_embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
            verbose=True
        )

        logger.info("Created Chroma vector store")
        vector_store.persist()
        logger.info("Persisted Chroma vector store")
