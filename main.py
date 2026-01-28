import plotly.express as px
import pandas as pd
import streamlit as st
import os
import warnings
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from numpy.random import default_rng as rng
from streamlit_molstar import st_molstar, st_molstar_rcsb, st_molstar_remote

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
    #data_pages = ["    📦 Models", "    💾 Datasets", "    🧪 Experiments"]
    #is_expanded = st.session_state.active_page in data_pages
    #with st.expander("📂 Data Pages", expanded=is_expanded):
        #for page in data_pages:
            #create_menu_item(page
    create_menu_item("📦 Models")
    create_menu_item("💾 Datasets")
    create_menu_item("🧪 Experiments")
    create_menu_item("📄 Generate Report")
    create_menu_item("🗺️ Map Search")
    
    st.markdown("---")
    create_menu_item("⚙️ Settings")
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

#elif "Models" in page:
    #st.title("Model Yönetimi")
    #st.info("Burada eğitilen OncoMind modelleri listelenecek.")

elif page == "📊 Charts":
    st.title("Grafikler")
    st.error("Bu sayfa hala yapım aşamasında ve gerçek olmayan test verileri kullanıyor!")
    col1, col2 = st.columns([1, 1], gap="small", border=True)
    with col1:
        st.bar_chart({"data": [10, 20, 30, 40]})
    with col2:
        df = px.data.iris()
        fig = px.scatter(
            df,
            x="sepal_width",
            y="sepal_length",
            color="species",
            size="petal_length",
            hover_data=["petal_width"],
        )

        event = st.plotly_chart(fig, key="iris", on_select="rerun")
    with st.container(border=True):
        st.subheader("Grafikler")
        
elif page == "🧠 Model Inferences":
    st.title("Model Çıkarımları")
    st.error("Bu sayfa hala yapım aşamasında ve gerçek olmayan test verileri kullanıyor!")
    hist_data = [
        rng(0).standard_normal(200) - 2,
        rng(1).standard_normal(200),
        rng(2).standard_normal(200) + 2,
    ]

    df = pd.DataFrame({
        "value": hist_data[0].tolist() + hist_data[1].tolist() + hist_data[2].tolist(),
        "group": (["Group 1"] * 200) + (["Group 2"] * 200) + (["Group 3"] * 200)
    })

    fig = px.histogram(
        df,
        x="value",
        color="group",
        marginal="rug",
        nbins=40,
        opacity=0.6
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "📦 Models":
    st.title("Model Yönetimi")
    st.error("Bu sayfa hala yapım aşamasında ve gerçek olmayan test verileri kullanıyor!")
    st.subheader("Modeller")
    st.info("Burada eğitilen OncoMind modelleri listelenecek.")
    for _ in range(3):  # rows
        cols = st.columns(4, border=True)
        for col in cols:
            with col:
                st.image("https://www.shutterstock.com/image-vector/ai-model-icon-designed-linear-600nw-2510832345.jpg")
                st.text("Model Adı")
                st.metric("Accuracy", "82%", "+12%")
                st.metric("Model Inference", "250 run/ms ", " -30 run/ms")
elif page == "🧪 Experiments":
    st.title("Deney Yönetimi")
    st.warning("Bu sayfa hala yapım aşamasında bu yüzden gerçek ve gerçek olmayan test verileri kullanıyor!")
    st.markdown("## 🧊 3D View")
    with st.container(border=True):
        st_molstar('molstar_examples/complex.pdb', 'molstar_examples/complex.xtc', key='4')
    st.markdown("## 📊 Analytics")

    col1, col2 = st.columns([2, 1], gap="small")
    mock_df = pd.DataFrame(
        [
            ["bob", "a"],
            ["sue", "b"],
            ["sue", "c"],
            ["joe", "c"],
            ["bill", "d"],
            ["max", "b"],
        ],
        columns=["A", "B"],
    )

    with col1:
        with st.container(border=True):
            st.subheader("DataFrame")
            st.dataframe(
                mock_df,
                use_container_width=True,
                height=350
            )

    with col2:
        with st.container(border=True):
            st.subheader("System Usage")
            st.metric("CPU Usage", "42%", "+3%") # Mock data
            st.metric("Memory", "8.1 GB", "-0.5 GB") # Mock data
    st.markdown("## 🗺️ Map")
    with st.container(border=True):
        # Mock data is used for testing
        df = pd.DataFrame(
            {
                "col1": 41.012764805965055,
                "col2": 28.948423970871268,
                "col3": rng(2).standard_normal(10000) * 100,
                "col4": rng(3).standard_normal((10000, 4)).tolist(),
            }
        )   
        st.map(df, latitude="col1", longitude="col2", size="col3", color="col4")
elif page == "📄 Generate Report":
    # Working Page
    st.title("Rapor Yönetim Merkezi 📄")
    if "last_report" not in st.session_state or not st.session_state.last_report:
        st.warning("⚠️ Henüz aktif bir rapor bulunmuyor.")
        st.info("Yeni bir rapor oluşturmak için '🏠 Home' sayfasına gidip bir araştırma başlatın.")
        if st.button("Ana Sayfaya Git", type="primary"):
            st.session_state.active_page = "🏠 Home"
            st.rerun()
            
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📝 Rapor Önizlemesi")
            with st.container(border=True):
                st.markdown(st.session_state.last_report)
        
        with col2:
            st.subheader("İşlemler")
            st.write("Raporu dışa aktar veya yönet.")
            st.download_button(
                label="📥 İndir (.txt)",
                data=st.session_state.last_report,
                file_name="arastirma_raporu.txt",
                mime="text/plain",
                use_container_width=True
            )
            if st.button("🗑️ Raporu Sil", type="primary", use_container_width=True):
                st.session_state.last_report = None
                st.success("Rapor hafızadan silindi.")
                st.rerun()

elif page == "🗺️ Map Search":
    # Mock data is used for testing
    st.warning("Bu sayfa hala yapım aşamasında ve gerçek olmayan test verileri kullanıyor!")
    st.title("Harita Araması 🗺️")
    st.subheader("Coğrafi Kanser Hastası Veri Analizi")
    df = pd.DataFrame(
        {
            "col1": 41.012764805965055,
            "col2": 28.948423970871268,
            "col3": rng(2).standard_normal(10000) * 100,
            "col4": rng(3).standard_normal((10000, 4)).tolist(),
        }
    )

    st.map(df, latitude="col1", longitude="col2", size="col3", color="col4")

else:
    st.title(f"{page}")
    st.write("Bu sayfa yapım aşamasında...")