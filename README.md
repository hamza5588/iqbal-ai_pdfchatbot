# PDF Reader LLM Flask App (Dockerized)

This project is a Flask web application for uploading PDF files and asking questions using LLMs and vector search. It is fully containerized for easy deployment.

## Features
- Upload and process PDF files
- Ask questions about uploaded documents using LLM
- Uses LangChain, ChromaDB, HuggingFace, and OpenAI integrations

## Getting Started

### Prerequisites
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/)

### Build and Run with Docker Compose

1. **Clone the repository** (if not already):
   ```sh
   git clone <your-repo-url>
   cd pdf_reader_llm_app_flask
   ```

2. **(Optional) Update your OpenAI API key**
   - Edit `api_key.txt` and put your OpenAI API key inside.

3. **Build and start the app:**
   ```sh
   docker-compose up --build
   ```
   The app will be available at [http://localhost:5000](http://localhost:5000)

4. **Upload PDFs and interact with the chatbot via the web UI.**

### Stopping the App
To stop the app, press `Ctrl+C` in the terminal, then run:
```sh
docker-compose down
```

### Notes
- Uploaded files and templates are mounted as volumes for persistence and development.
- To update dependencies, modify `requirements.txt` and rebuild the container.

## File Structure
- `app.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `templates/` - HTML templates
- `uploads/` - Uploaded PDF files
- `api_key.txt` - Your OpenAI API key (do not share this file)
- `Dockerfile` - Docker build instructions
- `docker-compose.yml` - Multi-container orchestration

---

**Security Note:** Never commit your real API keys to public repositories. 