import os
import warnings
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage

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

print("--- Research Agent Başlatılıyor (Vertex AI) ---")
inputs = {"task": "Unreal Engine 5 Nanite teknolojisi nedir?"}

for output in app.stream(inputs):
    pass

final_output = output["writer"]["final_report"]
print(f"\n📄 [NİHAİ RAPOR]:\n{final_output}")