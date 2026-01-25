import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Lütfen .env dosyasına GOOGLE_API_KEY ekle!")

# 2. Tool Tanımlama
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city} şehrinde hava şu an güneşli ve 25 derece!"

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)
agent = create_react_agent(model, tools=[get_weather])

print("--- Ajan Başlatılıyor (API Key Modu) ---")

inputs = {"messages": [HumanMessage(content="İstanbul'da hava nasıl?")]}

for chunk in agent.stream(inputs, stream_mode="values"):
    message = chunk["messages"][-1]
    if message.type == "ai":
        print(f"[AI]: {message.content}")
    elif message.type == "tool":
        print(f"[Tool Çıktısı]: {message.content}")