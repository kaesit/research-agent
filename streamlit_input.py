import streamlit as st
import os
import warnings
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

# --- 1. SAYFA KONFIGÜRASYONU ---
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Uyarıları kapat
warnings.filterwarnings("ignore")

# --- 2. STATE (DURUM) YÖNETİMİ ---
# Hangi sayfadayız ve son rapor neydi hafızada tutalım
if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Home"
if "last_report" not in st.session_state:
    st.session_state.last_report = None

# Performans için modeli cache'liyoruz, her seferinde tekrar bağlanmasın
@st.cache_resource
def get_graph():
    llm = ChatVertexAI(
        model="gemini-2.5-flash", 
        temperature=0
    )

    class AgentState(TypedDict):
        task: str
        research_result: str
        final_report: str

    def researcher(state: AgentState):
        task = state["task"]
        prompt = f"Şu konu hakkında 3 maddelik kısa ve teknik bilgi ver: {task}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"research_result": response.content}

    def writer(state: AgentState):
        research_data = state["research_result"]
        prompt = f"""
        Aşağıdaki teknik verileri kullanarak profesyonel, Türkçe bir özet rapor yaz.
        Veri: {research_data}
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_report": response.content}

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher)
    workflow.add_node("writer", writer)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)
    
    return workflow.compile()

app = get_graph()

def create_menu_item(label):
    btn_type = "primary" if st.session_state.active_page == label else "secondary"
    if st.button(label, key=label, type=btn_type, use_container_width=True):
        st.session_state.active_page = label
        st.rerun()

with st.sidebar:
    st.header("MENU")
    
    create_menu_item("🏠 Home")
    create_menu_item("📊 Charts")
    create_menu_item("🧠 Model Inferences")
    data_pages = ["    📦 Models", "    💾 Datasets", "    🧪 Experiments"]
    is_expanded = st.session_state.active_page in data_pages
    #with st.expander("📂 Data Pages", expanded=is_expanded):
        #for page in data_pages:
            #create_menu_item(page
            
    create_menu_item("📄 Generate Report")
    create_menu_item("🗺️ Map Search")
    
    st.markdown("---")
    create_menu_item("🔒 Authentication")
    create_menu_item("🚪 Logout")

page = st.session_state.active_page

if page == "🏠 Home":
    st.title("AI Research Agent 🕵️‍♂️")
    st.write("Google Vertex AI tarafından desteklenen araştırma asistanı.")
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Araştırma Konusu:", placeholder="Örn: Quantum Computing gelişmeleri...")
    if st.button("🚀 Araştırmayı Başlat", type="primary", use_container_width=True):
        if topic:
            with st.status("Agentlar Çalışıyor...", expanded=True) as status:
                st.write("🔎 Araştırmacı: Veri topluyor...")
                inputs = {"task": topic}
                
                # Agent Akışı
                final_res = None
                for output in app.stream(inputs):
                    for key, value in output.items():
                        st.write(f"✅ {key} görevi tamamladı.")
                        if "final_report" in value:
                            final_res = value["final_report"]
                
                st.session_state.last_report = final_res
                status.update(label="Araştırma Tamamlandı!", state="complete", expanded=False)
        else:
            st.warning("Lütfen bir konu giriniz.")

    if st.session_state.last_report:
        st.divider()
        st.subheader("📄 Final Rapor")
        st.markdown(st.session_state.last_report)

elif "Models" in page:
    st.title("Model Yönetimi")
    st.info("Burada eğitilen OncoMind modelleri listelenecek.")

elif page == "📊 Charts":
    st.title("Grafikler")
    st.bar_chart({"data": [10, 20, 30, 40]})

else:
    st.title(f"{page}")
    st.write("Bu sayfa yapım aşamasında...")