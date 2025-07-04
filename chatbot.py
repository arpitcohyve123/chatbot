import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import google.generativeai as genai

# Load your Gemini API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in a .env file.")

# Configure Gemini SDK for direct enhancement
genai.configure(api_key=API_KEY)

# Step 1: Load the document
def load_document(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found: {filepath}")
    except Exception as e:
        raise Exception(f"❌ Error reading file: {e}")

# Step 2: Create vector store using Gemini Embeddings
def create_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )

    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    return vectordb

# Step 3: Answer using Gemini model and vector search
def answer_query(vectordb, query):
    retriever = vectordb.as_retriever()
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=API_KEY,
        temperature=0.2
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain.run(query)

# Step 4: Enhance response directly with Gemini SDK
def gemini_enhance(text):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([
        {"role": "user", "parts": [f"Here is a document-based answer: {text}\nCan you elaborate or improve it?"]}
    ])
    return response.text

# Step 5: CLI Chat Loop
def main():
    print("📄 Loading and processing document...")

    try:
        doc_text = load_document("document.txt")
        vectordb = create_vector_store(doc_text)
        print("✅ Ready! Ask your questions (type 'exit' to quit).\n")

        while True:
            query = input("❓ Your question: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            try:
                print("\n🔎 Searching document...")
                answer = answer_query(vectordb, query)
                print(f"\n📘 Answer from document:\n{answer}\n")

                print("✨ Enhancing with Gemini...")
                enhanced = gemini_enhance(answer)
                print(f"\n💡 Gemini-enhanced response:\n{enhanced}\n")
                print("-" * 60)

            except Exception as e:
                print(f"⚠️ Error during query processing: {e}")

    except Exception as e:
        print(f"🚫 Initialization failed: {e}")

if __name__ == "__main__":
    main()
