# Environment Setup Instructions

## Prerequisites
- Miniconda installed
- Git installed
- Groq API key (free from https://console.groq.com)

## Installation Steps

1. Clone repository:
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/ci-migration-rag.git
cd ci-migration-rag
\`\`\`

2. Create conda environment:
\`\`\`bash
conda env create -f environment.yml
\`\`\`

3. Activate environment:
\`\`\`bash
conda activate ci_rag_env
\`\`\`

4. Create .env file with your API key:
\`\`\`
GROQ_API_KEY=your_groq_api_key_here
VECTORDB_PATH=./vectordb
\`\`\`

5. Place your scraped documentation in `data/raw/`

6. Build vector database:
\`\`\`bash
python src/prepare_vectordb.py
\`\`\`

7. Test the agent:
\`\`\`bash
python test_rag.py
\`\`\`
