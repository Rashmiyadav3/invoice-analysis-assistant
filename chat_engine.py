from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

def get_conversational_chain(api_key):
    """
    Sets up the QA Chain with Gemini Pro.
    """
    prompt_template = """
    You are an expert financial AI assistant analyzing an invoice.
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "The information is not available in the invoice context", don't provide a wrong answer.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=api_key
)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def answer_user_question(user_question, vector_store, chain):
    """
    Given a question and vector store, retrieve context and generate an answer.
    """
    # Retrieve top 4 most similar chunks to the question
    docs = vector_store.similarity_search(user_question, k=4)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]
