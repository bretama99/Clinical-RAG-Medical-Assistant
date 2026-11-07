# Clinical RAG Assistant

Efficient medical knowledge assistant with RAG and agentic workflow.

## Features

- **Simple RAG**: Retrieve from local PDFs and generate answers
- **Agentic RAG**: Decompose → Retrieve → Critique → Improve
- **Web Search**: Whitelisted medical domain fallback
- **Answer Verification**: Evidence-based claim checking

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your settings

# Place PDFs in data/raw/
mkdir -p data/raw
# Copy your medical PDFs here

# Run UI
streamlit run app/ui_streamlit.py
```

## Configuration

Edit `config.yaml` to customize:
- Model settings
- Retrieval parameters
- Web search whitelist
- Verification thresholds

## Project Structure

```
medical_rag/
├── src/
│   ├── agents/          # Multi-agent workflow
│   ├── llm/             # Ollama client
│   ├── prompting/       # System prompts
│   ├── rag/             # RAG pipeline & verifier
│   └── utils/           # Config & helpers
├── app/                 # Streamlit UI
├── data/
│   ├── raw/             # Input PDFs
│   └── processed/       # Index storage
└── config.yaml          # Configuration
```

## Requirements

- Python 3.10-3.13
- Ollama (running locally)
- Optional: Tavily API key for web search
