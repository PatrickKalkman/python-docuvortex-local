from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HuggingFacePipeline

from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores.chroma import Chroma

import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline

from settings import COLLECTION_NAME, PERSIST_DIRECTORY


class VortexQuery:
    def __init__(self):
        load_dotenv()
        self.chain = self.make_chain()
        self.chat_history = []

    def make_chain(self):

        # embeddings_model_name = ''
        # model_type = ''
        model_path = ''
        model_n_ctx = ''
        # target_source_chunks = 10000

        callbacks = [StreamingStdOutCallbackHandler()]

        tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")

        model = LlamaForCausalLM.from_pretrained("TheBloke/wizardLM-7B-HF",
                                                      load_in_8bit=True,
                                                      device_map='auto',
                                                      torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True
                                                      )

        pipe = pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=1024,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)

        instructor_embeddings = HuggingFaceInstructEmbeddings(
                                  model_name="hkunlp/instructor-xl",
                                  model_kwargs={"device": "cuda"})

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=instructor_embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        return RetrievalQA.from_chain_type(llm=local_llm, 
                                          chain_type="stuff", 
                                          retriever=retriever, 
                                          return_source_documents=True)

    def ask_question(self, question: str):
        response = self.chain(question)

        return response


