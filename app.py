from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_nomic import NomicEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

history = []

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store the chain and current files
retriever = None
current_files = []  # List to store multiple files
vector_store = None  # Global vector store for all documents

# Initialize LLM and embeddings


# API key storage
API_KEY_FILE = 'api_key.txt'

def load_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as f:
            return f.read().strip()
    else:
        return None

def save_api_key(new_key):
    with open(API_KEY_FILE, 'w') as f:
        f.write(new_key)

# Load API key on startup (optional, can be removed)
# load_api_key()

def get_llm():
    api_key = load_api_key()  # Always reload the API key from file
    return ChatOpenAI(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        openai_api_key=api_key
    )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def process_pdf(file_path, filename):
    """Process PDF and add to existing vector store or create new one"""
    global vector_store, retriever
    
    try:
        # Load PDF
        pdf_loader = PyPDFLoader(file_path)
        documents = pdf_loader.load()
        
        # Add filename metadata to documents
        for doc in documents:
            doc.metadata['source_file'] = filename
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200    
        )
        chunks = splitter.split_documents(documents)
        
        # Create or update vector store
        if vector_store is None:
            vector_store = Chroma.from_documents(chunks, embeddings)
        else:
            vector_store.add_documents(chunks)
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        return True
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False

@app.route('/')
def index():
    global current_files, retriever, vector_store, history
    # Delete all files in the uploads folder
    upload_folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    # Reset globals
    current_files = []
    retriever = None
    vector_store = None
    history = []
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_files, history
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a PDF file.'})
        
        filename = secure_filename(file.filename)
        
        # Check if file already exists
        if filename in current_files:
            return jsonify({'success': False, 'error': 'File already uploaded'})
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Process the PDF
        if process_pdf(file_path, filename):
            current_files.append(filename)
            return jsonify({
                'success': True, 
                'filename': filename,
                'total_files': len(current_files),
                'all_files': current_files
            })
        else:
            # Clean up failed file
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({'success': False, 'error': 'Error processing PDF file. Please try again.'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})

@app.route('/ask', methods=['POST'])
def ask_question():
    global retriever, history
    if retriever is None:
        return jsonify({'success': False, 'error': 'No PDF file processed yet'})

    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'success': False, 'error': 'No question provided'})

    try:
        # Build the chain on the fly with the latest LLM
        def retriever_func(question):
            global history
            history.append(HumanMessage(content=question))
            retrieved_docs = retriever.invoke(question)
            return {"context": retrieved_docs, "question": question, "history": history}

        retrieve_docs = RunnableLambda(retriever_func)
        parser = StrOutputParser()

        prompt = PromptTemplate(
            template="""
            You are an Intelligent AI that answers questions about Uploaded Documents.
            - Answer the question as truthfully as possible using the provided context.
            - If you can't find the answers in the context, just say that you don't know.
            - when the user says page no 1, he means page 0
            - when the user says page no 2, he means page 1, and so on
            - When referencing information, mention which document it comes from if relevant
            -Also do not mention page no until the user asks for it.

            Context: {context}

            Question: {question}

            Conversation History so far : {history}

            """,
            input_variables=["context", "question", "history"]
        )

        chain = retrieve_docs | prompt | get_llm() | parser
        result = chain.invoke(question)
        history.append(AIMessage(content=result))
        return jsonify({'success': True, 'answer': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    global current_files
    return jsonify({
        'has_file': len(current_files) > 0, 
        'files': current_files,
        'total_files': len(current_files)
    })

@app.route('/clear', methods=['POST'])
def clear_files():
    global current_files, retriever, vector_store, history
    
    try:
        # Clear uploaded files
        for filename in current_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                os.remove(file_path)
            except:
                pass
        
        # Clear vector store properly
        if vector_store is not None:
            try:
                # Delete the vector store collection
                vector_store.delete_collection()
            except:
                pass
        
        # Reset global variables
        current_files = []
        retriever = None
        vector_store = None
        history = []
        
        return jsonify({'success': True, 'message': 'All files cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    data = request.get_json()
    new_key = data.get('api_key', '').strip()
    if not new_key:
        return jsonify({'success': False, 'error': 'API key cannot be empty'})
    try:
        save_api_key(new_key)
        # Clear caches and objects that may hold the old API key
        global vector_store, retriever, history
        if vector_store is not None:
            try:
                vector_store.delete_collection()
            except:
                pass
        vector_store = None
        retriever = None
        history = []
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api_key_status')
def api_key_status():
    api_key = load_api_key()
    masked = api_key[:4] + '*' * (len(api_key)-8) + api_key[-4:] if api_key and len(api_key) > 8 else (api_key if api_key else None)
    return jsonify({'api_key': masked})

if __name__ == '__main__':
    app.run(debug=True)