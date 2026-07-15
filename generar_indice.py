import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Cargar variables de entorno
load_dotenv()


def generar_indice():

    carpeta_pdfs = "PDF"
    carpeta_vector = "vector_db"


    # Validar carpeta PDF
    if not os.path.exists(carpeta_pdfs):
        print(f"❌ La carpeta '{carpeta_pdfs}' no existe.")
        return


    print("=" * 70)
    print("       GENERADOR DEL ÍNDICE VECTORIAL - AGENTE IA QA")
    print("=" * 70)


    paginas = []
    cantidad_pdfs = 0


    # =====================================================
    # 1. CARGA DE DOCUMENTOS PDF
    # =====================================================

    print("\n📄 Leyendo Historias de Usuario...\n")


    for archivo in os.listdir(carpeta_pdfs):

        if archivo.lower().endswith(".pdf"):

            cantidad_pdfs += 1

            ruta = os.path.join(carpeta_pdfs, archivo)

            print(f"✔ Cargando: {archivo}")


            loader = PyPDFLoader(ruta)

            documentos = loader.load()


            # Guardar metadata del documento
            for doc in documentos:

                doc.metadata["HU"] = archivo
                doc.metadata["tipo_documento"] = "Historia de Usuario"


            paginas.extend(documentos)



    if len(paginas) == 0:
        print("\n❌ No se encontraron documentos PDF.")
        return



    print("\n----------------------------------")
    print(f"PDF encontrados: {cantidad_pdfs}")
    print(f"Total páginas cargadas: {len(paginas)}")
    print("----------------------------------")



    # =====================================================
    # 2. DIVISIÓN EN FRAGMENTOS
    # =====================================================

    print("\n✂️ Generando fragmentos...")


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )


    fragmentos = splitter.split_documents(paginas)


    print(f"Fragmentos generados: {len(fragmentos)}")



    # =====================================================
    # 3. CREACIÓN DE EMBEDDINGS LOCALES
    # =====================================================

    print("\n🧠 Generando embeddings locales...")


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



    # =====================================================
    # 4. CREACIÓN DEL ÍNDICE VECTORIAL FAISS
    # =====================================================

    print("\n🔎 Creando base vectorial FAISS...")


    vector_store = FAISS.from_documents(
        fragmentos,
        embeddings
    )



    # =====================================================
    # 5. GUARDAR ÍNDICE
    # =====================================================


    # Eliminar índice anterior si existe
    if os.path.exists(carpeta_vector):

        print("\n🗑 Eliminando índice anterior...")

        shutil.rmtree(carpeta_vector)



    vector_store.save_local(carpeta_vector)



    print("\n" + "=" * 70)
    print("✅ ÍNDICE VECTORIAL CREADO CORRECTAMENTE")
    print("=" * 70)

    print(f"""
    📁 Ubicación:
       {carpeta_vector}

    📄 Documentos procesados:
       {cantidad_pdfs}

    📚 Páginas:
       {len(paginas)}

    🧩 Fragmentos:
       {len(fragmentos)}

    """)



if __name__ == "__main__":
    generar_indice()