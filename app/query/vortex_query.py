from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores.chroma import Chroma

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

        llm = GPT4All(model=model_path, 
                      n_ctx=model_n_ctx, 
                      backend='gptj', 
                      callbacks=callbacks, verbose=False)

        instructor_embeddings = HuggingFaceInstructEmbeddings(
                                  model_name="hkunlp/instructor-xl",
                                  model_kwargs={"device": "cpu"})

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=instructor_embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

        return ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
        )

    def ask_question(self, question: str):
        response = self.chain({"question": question, "chat_history": self.chat_history})

        answer = response["answer"]
        source = response["source_documents"]
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer, source
