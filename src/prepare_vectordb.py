"""
Vector Database Preparation Module
Converts text documentation to ChromaDB vector database
"""
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class DocumentProcessor:
    """Process and vectorize CodeIgniter documentation"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        print("🔧 Initializing embedding model...")
        # Free, lightweight, and effective for technical docs
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
        )
        print("✅ Embedding model loaded")
        
    def load_documents(self) -> List:
        """Load all text files from data directory"""
        documents = []
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory {self.data_dir} not found. Please create it and add your .txt files")
        
        txt_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.data_dir}")
        
        print(f"\n📚 Loading {len(txt_files)} documentation files...")
        
        for filename in txt_files:
            file_path = os.path.join(self.data_dir, filename)
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                # Add metadata to identify source
                for doc in docs:
                    doc.metadata['source_file'] = filename
                    doc.metadata['doc_type'] = self._identify_doc_type(filename)
                
                documents.extend(docs)
                print(f"  ✓ {filename}: {len(docs)} documents")
                
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        
        print(f"\n✅ Total documents loaded: {len(documents)}")
        return documents
    
    def _identify_doc_type(self, filename: str) -> str:
        """Identify document type from filename"""
        filename_lower = filename.lower()
        
        if 'ci3' in filename_lower and 'ci4' not in filename_lower:
            return 'ci3_documentation'
        elif 'ci4' in filename_lower:
            return 'ci4_documentation'
        elif 'upgrade' in filename_lower or 'migration' in filename_lower:
            return 'upgrade_guide'
        elif 'model' in filename_lower:
            return 'model_docs'
        elif 'view' in filename_lower:
            return 'view_docs'
        elif 'helper' in filename_lower:
            return 'helper_docs'
        elif 'library' in filename_lower or 'lib' in filename_lower:
            return 'library_docs'
        
        return 'general'
    
    def split_documents(self, documents: List) -> List:
        """Split documents into chunks for better retrieval"""
        print("\n✂️  Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust based on your docs
            chunk_overlap=200,  # Maintains context between chunks
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        return chunks
    
    def create_vectordb(self, chunks: List, persist_directory: str = "vectordb") -> Chroma:
        """Create and persist vector database"""
        print(f"\n🔨 Creating vector database in '{persist_directory}'...")
        print("⏳ This may take 2-5 minutes depending on document size...")
        
        # Create directory if doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_directory,
                collection_name="ci_migration_docs"
            )
            
            count = vectordb._collection.count()
            print(f"\n✅ Vector database created successfully!")
            print(f"   📊 Total vectors: {count}")
            print(f"   📁 Location: {persist_directory}")
            
            return vectordb
            
        except Exception as e:
            print(f"\n❌ Error creating vector database: {e}")
            raise

def main():
    """Main execution function"""
    print("="*60)
    print("CodeIgniter Migration RAG - Vector Database Builder")
    print("="*60)
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        
        # Load documents
        docs = processor.load_documents()
        
        # Split into chunks
        chunks = processor.split_documents(docs)
        
        # Create vector database
        vectordb = processor.create_vectordb(chunks)
        
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE! Your RAG agent is ready to use.")
        print("="*60)
        print("\nNext steps:")
        print("1. Test the agent: python test_rag.py")
        print("2. Integrate with your migrator: see src/integration.py")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\nTroubleshooting:")
        print("- Ensure your .txt files are in data/raw/")
        print("- Check file encoding (should be UTF-8)")
        print("- Verify conda environment is activated")

if __name__ == "__main__":
    main()
