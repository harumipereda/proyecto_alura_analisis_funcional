import os
from dotenv import load_dotenv

# Módulos estables de LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar la API Key desde el archivo .env
load_dotenv()


def iniciar_agente():
    carpeta_pdfs = "PDF"
    
    if not os.path.exists(carpeta_pdfs):
        print(f"Error: La carpeta '{carpeta_pdfs}' no existe.")
        return

    paginas_totales = []
    print("Iniciando la lectura de documentos...")
    print("="*60)
    print("AGENTE IA PARA ANÁLISIS FUNCIONAL")
    print("="*60)
    for archivo in os.listdir(carpeta_pdfs):
        if archivo.endswith(".pdf"):
            ruta_completa = os.path.join(carpeta_pdfs, archivo)
            print(f" Cargando: {archivo}")
            loader = PyPDFLoader(ruta_completa)
            paginas_totales.extend(loader.load())

    print(f"\n¡Éxito! Se cargaron {len(paginas_totales)} páginas.")

    if len(paginas_totales) == 0:
        print("No se encontraron archivos PDF para procesar.")
        return

    # 1. Fragmentación del texto
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100)
    fragmentos = text_splitter.split_documents(paginas_totales)
    print(f"Páginas cargadas: {len(paginas_totales)}")
    print(f"Fragmentos generados: {len(fragmentos)}")

    # 2. Creación del almacén de vectores (FAISS)
    print("Indexando documentos en la base de datos vectorial...")
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_documents(fragmentos, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 3. Configuración del modelo Gemini
    llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"))

    # 4. Diseño del Prompt estructurado
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    Eres un asistente especializado en análisis funcional y pruebas funcionales.

    Tus conocimientos provienen únicamente de las Historias de Usuario proporcionadas.

    Puedes:
    - responder preguntas
    - explicar reglas de negocio
    - identificar actores
    - resumir historias
    - generar casos de prueba funcionales
    - identificar validaciones
    - identificar dependencias entre historias

    Si la respuesta no está en los documentos responde exactamente:

    "No encontré esa información en los documentos proporcionados."

    Responde de forma clara, profesional y utilizando únicamente el contexto proporcionado.

    Contexto:
    {context}
    """),
    ("human", "{question}")
    ])

    # Función auxiliar para formatear los documentos encontrados por el retriever
    def formatear_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Construcción de la cadena RAG usando LCEL (Sin chains obsoletas)
    rag_chain = (
        {"context": retriever | formatear_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nPuedes realizar preguntas como:")

    print("- ¿Cuál es el objetivo de la HU?")
    print("- ¿Qué reglas de negocio existen?")
    print("- ¿Qué actores participan?")
    print("- Resume la historia.")
    print("- Genera casos de prueba.")
    print("- ¿Qué validaciones existen?")
    print("- Escribe 'salir' para terminar.\n")

    # 6. Ejecución de la prueba
   
    while True:
        pregunta = input("\nEscribe tu pregunta: ")

        if pregunta.lower() == "salir":
            print("\nGracias por usar el asistente.")
            break

        print("\nBuscando información...\n")

        respuesta = rag_chain.invoke(pregunta)

        print("\nRespuesta del agente:\n")
        print(respuesta)


if __name__ == "__main__":
    iniciar_agente()