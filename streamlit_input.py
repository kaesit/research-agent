import streamlit as st
import streamlit_antd_components as sac
import os
import warnings
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(
    page_title="Agent Research App",
    page_icon="🔬",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

warnings.filterwarnings("ignore")
llm = ChatVertexAI(
    model="gemini-2.5-flash",
    temperature=0
)

class AgentState(TypedDict):
    task: str
    research_result: str
    final_report: str

def researcher(state: AgentState):
    print("\n🔎 [Araştırmacı]: Konu hakkında bilgi topluyorum...")
    task = state["task"]
    
    prompt = f"Şu konu hakkında 3 maddelik kısa ve teknik bilgi ver: {task}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"research_result": response.content}

def writer(state: AgentState):
    print("✍️  [Yazar]: Raporu yazıyorum...")
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

app = workflow.compile()
st.title("AI Research Agent")

with st.sidebar:
    selected_item = sac.menu([
        sac.MenuItem('Home', icon='house'),
        sac.MenuItem('Charts', icon='bar-chart'),
        sac.MenuItem('Model Inferences', icon='cpu'),
        sac.MenuItem('Data Pages', icon='database', children=[
            sac.MenuItem('Models', icon='box'),
            sac.MenuItem('Datasets', icon='table'),
            sac.MenuItem('Experiments', icon='card-checklist'),
        ]),
        
        sac.MenuItem('Generate Report', icon='file-text'),
        sac.MenuItem('Map Search', icon='map'),
        sac.MenuItem("Authentication", icon='fingerprint'),
        sac.MenuItem('Settings', icon='gear'),
        sac.MenuItem(type='divider'), 
        sac.MenuItem('Logout', icon='box-arrow-right'),
    ], open_all=False) # open_all=False ile kapalı gelir, tıklayınca açılır.
st.write("Welcome to the AI Research Agent powered by Vertex AI. Please enter your research topic below.")
topic = st.text_input("Enter your research topic:")
if topic:
    st.success(f"Research topic set to: {topic}")

print("--- Research Agent Başlatılıyor (Vertex AI) ---")
inputs = {"task": topic}

for output in app.stream(inputs):
    pass

final_output = output["writer"]["final_report"]
st.header("Final Report")
st.write(final_output)