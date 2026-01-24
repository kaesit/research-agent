from langchain.agents import create_agent
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    task: str
    research_data: List[str]
    report: str
    current_status: str

def research_node(state: AgentState):
    print(f"--- [Researcher] Çalışıyor: {state['task']} ---")
    
    yeni_veri = "Source 2 motoru C++ ile yazılmıştır ve Vulkan API kullanır."
    
    return {
        "research_data": [yeni_veri],
        "current_status": "research_done"
    }

def writing_node(state: AgentState):
    print(f"--- [Writer] Çalışıyor ---")

    veriler = state['research_data']
    
    rapor = f"RAPOR: Toplanan verilere göre: {veriler[0]}"
    
    return {
        "report": rapor,
        "current_status": "writing_done"
    }

workflow = StateGraph(AgentState)

workflow.add_node("researcher", research_node)
workflow.add_node("writer", writing_node)


workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

inputs = {"task": "Source 2 oyun motorunu araştır", "research_data": [], "report": "", "current_status": "start"}

for output in app.stream(inputs):
    for key, value in output.items():
        print("----")
        print(f"Biten Düğüm: {key}")
        print(f"Güncel State: {value}")