import csv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key: str = os.getenv("OPENAI_API_KEY")

def read_from_csv(csv_path: str) -> list[Document]:
    """
    Reads the CSV file and converts its content into a list of Document objects.

    :param csv_path: The path to the CSV file.
    :return: A list of Document objects.
    """
    documents = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            doc = Document(
                page_content=row['page_content'],
                metadata={
                    "page_number": int(row['page_number']),
                    "chunk": int(row['chunk']),
                    "source": row['source'],
                    # Add other metadata fields if needed
                }
            )
            documents.append(doc)
    return documents

if __name__ == "__main__":
    # Step 1: Read from CSV
    csv_path = "processed_pdf.csv"
    document_chunks = read_from_csv(csv_path)

    # Step 3 + 4: Generate embeddings and store them in DB
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name="manus",
        persist_directory="chroma",
    )

    # Save DB locally
    vector_store.persist()

    # Number of documents in the vector store and DB written 
    print(f"Number of Document Chunks: {len(document_chunks)}")

    

