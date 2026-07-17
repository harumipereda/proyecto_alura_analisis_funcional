import os
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno
load_dotenv()

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Agente IA - Análisis Funcional SEACE",
    page_icon="🤖",
    layout="wide"
)

# Título de la aplicación web
st.title("🤖 Agente IA de Análisis Funcional (SEACE)")
st.subheader("Consulta reglas de negocio, historias de usuario y casos de prueba en segundos.")

# =====================================================
# Carga de Base Vectorial (Usando caché de Streamlit para que no cargue en cada clic)
# =====================================================
@st.cache_resource
def inicializar_motor_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    

    llm = ChatOllama(
        model="llama3.1",
        temperature=0.2,
         num_predict=512
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Eres un asistente experto en análisis funcional y pruebas funcionales de software.
        Tu conocimiento proviene únicamente de las Historias de Usuario cargadas.

        Puedes:
        - Explicar objetivos de Historias de Usuario.
        - Identificar reglas de negocio.
        - Identificar actores.
        - Explicar escenarios.
        - Generar casos de prueba funcionales.
        - Identificar validaciones.
        - Relacionar Historias de Usuario.

        Cuando compares Historias de Usuario:
        - identifica similitudes.
        - identifica diferencias.
        - identifica dependencias.
        - menciona las reglas de negocio relacionadas.

        Cada documento corresponde a una Historia de Usuario. Usa el nombre del documento cuando sea relevante.

        Si no encuentras información responde:
        "No encontré esa información en los documentos proporcionados."

        Contexto:
        {context}
        """),
        ("human", "{question}")
    ])
    
    def formatear_docs(docs):
        resultado = ""
        for doc in docs:
            hu = doc.metadata.get("HU", "Documento sin nombre")
            resultado += f"\n\nHistoria de Usuario: {hu}\n{doc.page_content}"
        return resultado

    rag_chain = (
        {"context": retriever | formatear_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Inicializar el motor RAG
try:
    rag_chain = inicializar_motor_rag()
    st.success("✅ Base de conocimientos e IA listas para responder.")
except Exception as e:
    st.error(f"❌ Error al cargar la base de conocimientos o conectar con el modelo IA: {e}")
    st.stop()

# =====================================================
# Historial de Chat interactivo en pantalla
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if pregunta := st.chat_input("¿Qué deseas consultar sobre los documentos de SEACE?"):
    # Mostrar la pregunta del usuario en la interfaz
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Generar y mostrar la respuesta de la IA
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos y redactando respuesta..."):
            respuesta = rag_chain.invoke(pregunta)
            st.markdown(respuesta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})