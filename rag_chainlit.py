import chainlit as cl
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from embedding_helper import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Don't answer as you are assuming answer as you are assistant
Answer as a support engineer. for a company infraon and its product is like it inventory management system types so..
from route_database.pdf form data file or embeddings from chroma files pdf/csv file is there in training or embeded data so provide respective route or url link whaen specific questions asked like add ticket or edit ticket etc. as aseprate if you want this is the url or follow this url like that 
Answer the question based on the above context: {question}
"""

# Prepare database + LLM once at startup
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
model = OllamaLLM(model="llama3.2")


def query_rag(query_text: str):
    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Build context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Get response
    response_text = model.invoke(prompt)

    # Collect metadata sources
    sources = [doc.metadata.get("id", None) for doc, _ in results]

    return response_text, sources


@cl.on_message
async def main(message: cl.Message):
    query_text = message.content
    response_text, sources = query_rag(query_text)

    # Send model response to UI
    await cl.Message(content=f"**Response:**\n{response_text}").send()

    # Show document sources separately
    if sources:
        await cl.Message(
            content="**Document Embeddings Referred:**\n" + "\n".join([f" - {s}" for s in sources if s])
        ).send()
