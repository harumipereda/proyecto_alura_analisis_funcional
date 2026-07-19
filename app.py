import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# Cargar variables de entorno
# =====================================================

load_dotenv()

# Si no está en el .env, intentamos buscarla en los Secrets de Streamlit
if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Validar que finalmente exista la clave en el sistema
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ No se configuró la variable GOOGLE_API_KEY. Verifica los Secrets en Streamlit o tu archivo .env local.")
    st.stop()


# =====================================================
# Configuración de Streamlit
# =====================================================

st.set_page_config(
    page_title="Agente IA - Análisis Funcional SECAE",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Agente IA de Análisis Funcional (SECAE)")
st.subheader(
    "Consulta reglas de negocio, historias de usuario y genera casos de prueba."
)


# =====================================================
# Inicialización del motor RAG
# =====================================================


def inicializar_motor_rag():

    # Embeddings locales
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Cargar base vectorial creada previamente
    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )


    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )


    # Modelo Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        max_output_tokens=512
    )


    # Prompt del agente
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Eres un asistente experto en análisis funcional
y pruebas funcionales de software.

Tu conocimiento proviene únicamente de las
Historias de Usuario cargadas en la base documental.

Puedes ayudar con:

- Explicar objetivos de Historias de Usuario.
- Identificar actores.
- Identificar reglas de negocio.
- Explicar escenarios.
- Identificar validaciones.
- Generar casos de prueba funcionales.
- Comparar Historias de Usuario.
- Identificar dependencias entre historias.

Cuando generes casos de prueba incluye:
- Caso de prueba.
- Precondiciones.
- Datos de prueba.
- Pasos.
- Resultado esperado.

Reglas importantes:

- No inventes información.
- Usa únicamente el contexto proporcionado.
- Si la información no está en los documentos responde:

"No encontré esa información en los documentos proporcionados."

Cada documento corresponde a una Historia de Usuario.
Utiliza el nombre del documento cuando sea relevante.

Contexto:
{context}

"""
            ),
            (
                "human",
                "{question}"
            )
        ]
    )


    # Formatear documentos recuperados

    def formatear_docs(docs):

        resultado = ""

        for doc in docs:

            hu = doc.metadata.get(
                "HU",
                "Documento sin nombre"
            )

            resultado += (
                f"\n\nHistoria de Usuario: {hu}\n"
                f"{doc.page_content}"
            )

        return resultado



    # Cadena RAG

    rag_chain = (
        {
            "context": retriever | formatear_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


    return rag_chain



# =====================================================
# Cargar motor
# =====================================================

try:

    rag_chain = inicializar_motor_rag()

    st.success(
        "✅ Base de conocimientos e IA listas para responder."
    )


except Exception as e:

    st.error(
        f"❌ Error al cargar el agente IA: {e}"
    )

    st.stop()



# =====================================================
# Historial del chat
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = []



for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



# =====================================================
# Entrada del usuario
# =====================================================

if pregunta := st.chat_input(
    "¿Qué deseas consultar sobre las Historias de Usuario?"
):


    st.session_state.messages.append(
        {
            "role": "user",
            "content": pregunta
        }
    )


    with st.chat_message("user"):

        st.markdown(pregunta)



    with st.chat_message("assistant"):


        with st.spinner(
            "Buscando información y generando respuesta..."
        ):


            respuesta = rag_chain.invoke(
                pregunta
            )


            st.markdown(respuesta)



    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta
        }
    )