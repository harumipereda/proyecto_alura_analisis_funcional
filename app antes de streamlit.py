import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


def iniciar_agente():

    print("\n====================================")
    print(" AGENTE IA DE ANÁLISIS FUNCIONAL")
    print("====================================")


    # =====================================================
    # 1. Cargar base vectorial creada previamente
    # =====================================================

    print("\nCargando base de conocimientos...")


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )


    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )


    print("Base de conocimientos cargada correctamente.")



    # =====================================================
    # 2. Configurar modelo generativo Gemini
    # =====================================================

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )



    # =====================================================
    # 3. Prompt del agente
    # =====================================================

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Eres un asistente experto en análisis funcional
                y pruebas funcionales de software.

                Tu conocimiento proviene únicamente de las
                Historias de Usuario cargadas.

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

                Cada documento corresponde a una Historia de Usuario.
                Usa el nombre del documento cuando sea relevante.

                Si no encuentras información responde:

                "No encontré esa información en los documentos proporcionados."

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



    # =====================================================
    # 4. Formatear documentos recuperados
    # =====================================================

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



    # =====================================================
    # 5. Construcción RAG
    # =====================================================

    rag_chain = (

        {
            "context": retriever | formatear_docs,
            "question": RunnablePassthrough()
        }

        | prompt
        | llm
        | StrOutputParser()

    )



    print("""
Puedes preguntar:

- ¿Cuál es el objetivo de la HU EC001?
- ¿Qué reglas de negocio tiene EC002?
- ¿Qué HU tienen validaciones con CRENIEC?
- Genera casos de prueba para esta historia.
- ¿Qué actores participan?
- ¿Qué historias están relacionadas?

Escribe 'salir' para terminar.
""")



    # =====================================================
    # 6. Interacción con usuario
    # =====================================================

    while True:

        pregunta = input("\nPregunta: ")


        if pregunta.lower() == "salir":

            print("Cerrando agente...")
            break


        print("\nBuscando información...\n")


        respuesta = rag_chain.invoke(pregunta)


        print("Respuesta:")
        print(respuesta)



if __name__ == "__main__":
    iniciar_agente()