from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = None

def getEmbedder():
    global embeddings
    if embeddings is None :
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    return embeddings
