import streamlit as st
import streamlit as st
import os
import PyPDF2
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import random
st.markdown("""
    <style>
        /* 1. Nasconde il link a GitHub e il tasto Deploy, ma SALVA i 3 puntini del menù */
      header a { display: none !important; }
      .stAppDeployButton { display: none !important; }

      /* 2. Sfondo di default (Tema Chiaro): Immagine chiara sfocata + velo bianco */
      .stApp::before {
          content: "";
          position: fixed;
          /* Lo facciamo leggermente più grande per non vedere i bordi sfocati */
          top: -5%; left: -5%; width: 110%; height: 110%; 
          z-index: -1;
          background-size: cover;
          background-position: center;
          filter: blur(8px); /* <--- IL LIVELLO DI SFOCATURA */
        
          /* Immagine diurna con velo bianco al 60% */
          background-image: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)), 
                          url("https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=2070&auto=format&fit=crop");
      }

    /* 3. Variante per il Tema Scuro: Immagine scura sfocata + velo nero/grigio */
      @media (prefers-color-scheme: dark) {
          .stApp::before {
              /* Immagine notturna con velo scuro all'85% per far leggere le scritte bianche */
              background-image: linear-gradient(rgba(14, 17, 23, 0.85), rgba(14, 17, 23, 0.85)), 
                              url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=2000&auto=format&fit=crop");
          }
    }
    </style>
    """, unsafe_allow_html=True)
# 1. SETUP INIZIALE
load_dotenv()
# Carica le chiavi
CHIAVI_DISPONIBILI = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2")
    ]
# Rimozione eventuali valori vuoti (per due chiavi)
CHIAVI_VALIDE = [k for k in CHIAVI_DISPONIBILI if k is not None]

def get_groq_client():
    """Restituisce un client Groq usando una chiave casuale dalla lista"""
    if not CHIAVI_VALIDE:
        st.error("Nessuna API Key trovata nel file .env!")
        return None
    chiave_scelta = random.choice(CHIAVI_VALIDE)
    return Groq(api_key=chiave_scelta)
# Inizializzazione del primo client (si aggiorna sempre) 
client = get_groq_client()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# 2. FUNZIONE MULTI-FORMATO (PDF + CSV) - Fase C
def processa_cartella(cartella):
    tutti_i_chunks = []
    if not os.path.exists(cartella):
        os.makedirs(cartella)
        
    for filename in os.listdir(cartella):
        path = os.path.join(cartella, filename)
        
        # Gestione PDF
        if filename.endswith(".pdf"):
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    testo = "".join([p.extract_text() for p in reader.pages])
                    for i in range(0, len(testo), 1000):
                        tutti_i_chunks.append(f"[{filename}]: " + testo[i:i+1000])
            except Exception as e:
                st.error(f"Errore leggendo PDF {filename}: {e}")
        
        # Gestione CSV (Fase C)
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(path)
                for index, row in df.iterrows():
                    frase_riga = f"In {filename}, riga {index}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
                    tutti_i_chunks.append(frase_riga)
            except Exception as e:
                st.error(f"Errore leggendo CSV {filename}: {e}")
                
    return tutti_i_chunks

# --- INTERFACCIA ---
st.set_page_config(page_title="DocTor", layout="wide")
st.title("🤖 Agente IA DocTor v2.0")
# --- GUIDA ALLE IA ---
with st.sidebar.expander("❓ Quale IA scegliere?"):
    st.markdown("""
    - **NomeModello*-8b**: Ultra-veloce, ottima per riassunti semplici e domande dirette.
    - **NomeModello*-70b**: Il giusto compromesso. Molto intelligente, brava nel ragionamento logico.
    - **NomeModello*-120b**: La più potente. Ideale per analisi complesse, dati tecnici e file CSV complicati.
    
    *Nota: I modelli più grandi potrebbero essere leggermente più lenti nel rispondere.*
    """)
# Sidebar
st.sidebar.header("⚙️ Configurazione")
try:
    lista_modelli = client.models.list()
    nomi_modelli = [m.id for m in lista_modelli.data]
    modello_scelto = st.sidebar.selectbox("🧠 Modello IA", nomi_modelli, index=0)
except:
    modello_scelto = "mixtral-8x7b-32768"

# --- CARICAMENTO FILE ---
st.sidebar.subheader("📁 Carica Documenti")
file_caricati = st.sidebar.file_uploader(
    "Trascina qui i tuoi PDF o CSV", 
    type=["pdf", "csv"], 
    accept_multiple_files=True
)
# Privacy
st.sidebar.caption("🔒 I file sono caricati solo temporaneamente e cancellati alla chiusura della sessione.")

if st.sidebar.button("🔄 Indicizza Documenti"):
    if file_caricati:
        with st.spinner("Analisi dei file in corso..."):
            tutti_i_chunks = []
            
            for file in file_caricati:
                # Se è un PDF
                if file.name.endswith(".pdf"):
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(file)
                    testo = ""
                    for page in pdf_reader.pages:
                        testo += page.extract_text()
                    # Dividi in pezzi da 1000 caratteri
                    for i in range(0, len(testo), 1000):
                        tutti_i_chunks.append(f"[{file.name}]: {testo[i:i+1000]}")
                
                # Se è un CSV
                elif file.name.endswith(".csv"):
                    import pandas as pd
                    df = pd.read_csv(file)
                    for index, row in df.iterrows():
                        contenuto_riga = ", ".join([f"{col}: {val}" for col, val in row.items()])
                        tutti_i_chunks.append(f"[{file.name} - riga {index}]: {contenuto_riga}")

            if tutti_i_chunks:
                # Crea i vettori e l'indice FAISS
                st.session_state.chunks = tutti_i_chunks
                vettori = embed_model.encode(tutti_i_chunks)
                
                import faiss
                import numpy as np
                dim = vettori.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(np.array(vettori))
                
                st.session_state.index = index
                st.sidebar.success(f"✅ Documentazione caricata con successo! Interrogatorio disponibile -->")
                st.balloons()
    else:
        st.sidebar.warning("⚠️ Carica almeno un file prima!")

st.sidebar.divider()
with st.sidebar.expander("ℹ️ Come funziona l'Agente?"):
    st.markdown("""
    Questo agente utilizza la tecnologia **RAG** (Retrieval-Augmented Generation):
    1. **Analisi**: Legge i file che carichi.
    2. **Memoria**: Crea un indice vettoriale dei contenuti.
    3. **Risposta**: Quando fai una domanda, l'IA cerca le parti rilevanti nei tuoi documenti e le usa per risponderti in modo preciso.
    """)
st.sidebar.divider() # Aggiunge una linea sottile di separazione
st.sidebar.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.8em;'>
        Powered by Marrasenzacash 🇮🇹
    </div>
    """, 
    unsafe_allow_html=True
)
# Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_utente := st.chat_input("Fai una domanda..."):
    st.session_state.messages.append({"role": "user", "content": prompt_utente})
    with st.chat_message("user"):
        st.markdown(prompt_utente)

    if st.session_state.index:
        # FASE A: Query Expansion
        with st.spinner("L'IA sta riflettendo..."):
            prompt_exp = f"Riformula la domanda '{prompt_utente}' in 3 termini tecnici chiave per la ricerca. Rispondi solo con i termini."
            res_exp = client.chat.completions.create(model=modello_scelto, messages=[{"role": "user", "content": prompt_exp}])
            query_estesa = prompt_utente + " " + res_exp.choices[0].message.content

        # Ricerca
        v_domanda = embed_model.encode([query_estesa])
        _, indici = st.session_state.index.search(np.array(v_domanda), k=4)
        contesto = "\n\n".join([st.session_state.chunks[i] for i in indici[0]])
        storia = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
        
        # Prompt Finale
        prompt_finale = f"Usa il contesto: {contesto}\n\nStoria chat: {storia}\n\nRispondi a: {prompt_utente}"
        
        with st.chat_message("assistant"):
            try:
            	#Prima riga aggiorna la chiave che utilizza(random)
                client = get_groq_client()
            
                response = client.chat.completions.create(
                    model=modello_scelto,
                    messages=[{"role": "system", "content": "Sei un assistente aziendale preciso."},
                              {"role": "user", "content": prompt_finale}],
                    temperature=0.1
                )
                risposta = response.choices[0].message.content
                st.markdown(risposta)
                st.session_state.messages.append({"role": "assistant", "content": risposta})
            except Exception as e:
                st.error(f"Errore: {e}")
    else:
        st.warning("⚠️ Indicizza i documenti prima!")
