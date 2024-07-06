import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    if "call me" in user_question.lower():
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def send_email(to_email, subject, body):
    sender_email=os.getenv("EMAIL_ADDRESS")
    sender_password=os.getenv("APP_PASSWORD")

    if not sender_email or not sender_password:
        st.error("Email credentials not set. Please set EMAIL_ADDRESS and APP_PASSWORD in the environment variables.")
        return False

    msg = MIMEText(body)

    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email,sender_password)
        server.sendmail(sender_email,to_email,msg.as_string())
        print("Email has been sent to "+ to_email)
        server.quit() 
        return True
    
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False


def main():
    st.set_page_config(page_title="ChatBot")
    st.header("🛸🛸Chat with PDF🛸🛸")

    user_question = st.text_input("Question your PDF Files")
    if user_question:
        response = user_input(user_question)
        
        if response is not None:
            st.write("Output:", response)
        
        if response is None and "call me" in user_question.lower():
            with st.form(key='user_info_form'):
                st.write("Please provide your contact information:")
                name = st.text_input("Name")
                phone = st.text_input("Phone Number")
                email = st.text_input("Email")
                send_email_option = st.checkbox("Send email")
                submit_button = st.form_submit_button(label='Submit')
                
                if submit_button:
                    st.write(f"Thank you, {name}. We will contact you at {phone}.")
                    if send_email_option:
                        subject = "Contact Request Confirmation"
                        body = f"Hello {name},\n\nThank you for providing your contact information. We will contact you at {phone} or via email at {email} shortly.\n\nBest regards,\nYour Company"
                        if send_email(email, subject, body):
                            st.success("Email sent successfully!")
                        else:
                            st.error("Failed to send email.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(" [1] Upload your PDF Files \n\n [2] Click on the Submit Button \n\n [3] Start your queries \n\n [4] Type 'call me' to get in contact", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing the File..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
