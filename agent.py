import streamlit as st
import io
import base64
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import io
import os
import sqlite3
import requests
from typing import Optional
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ---------------------- ENV SETUP ----------------------

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

DB_PATH = "incidents.db"

# ---------------------- DATABASE INIT ----------------------
def create_incidents_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            location TEXT NOT NULL,
            image_url TEXT,
            category TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

create_incidents_table()

# ---------------------- LLM & EMBEDDINGS ----------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.0
)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

vectordb = Chroma(
    persist_directory="chroma_store_gemini",
    embedding_function=embedding
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ---------------------- MEMORY ----------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",
    return_messages=True
)

# ---------------------- TOOLS ----------------------
@tool
def insert_incident_to_db(description: str, location: str, image_url: Optional[str] = None, category: Optional[str] = "hazard") -> str:
    """Insert a safety incident into the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO incidents (description, location, image_url, category)
            VALUES (?, ?, ?, ?)
        """, (description, location, image_url, category))
        conn.commit()
        conn.close()
        return "✅ Incident successfully logged."
    except Exception as e:
        return f"❌ DB error: {str(e)}"

@tool
def get_recent_incidents(limit: int = 5) -> str:
    """Fetch the latest incident reports."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT description, location, category, timestamp
            FROM incidents
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "ℹ️ No incidents found."
        return "\n\n".join([
            f"📍 {loc} | 🗂 {cat} | 🕒 {ts}\n📝 {desc}"
            for desc, loc, cat, ts in rows
        ])
    except Exception as e:
        return f"❌ Error fetching incidents: {str(e)}"

# @tool
# def report_safety_issue(description: str, location: str, image_url: Optional[str] = None) -> str:
#     """Store a user safety report into the database."""
#     categories = ["fire", "accident", "theft", "hazard"]
#     category = next((c for c in categories if c in description.lower()), "hazard")
#     return insert_incident_to_db.run(description=description, location=location, image_url=image_url, category=category)

@tool
def answer_safety_faq(query: str) -> str:
    """Answer public safety FAQs using retrieved knowledge."""
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = rag_chain.invoke({"query": query})
    answer = result["result"]
    sources = result["source_documents"]

    citations = []
    for i, doc in enumerate(sources, 1):
        source = doc.metadata.get("source", f"doc_{i}")
        page = doc.metadata.get("page", "N/A")
        excerpt = doc.page_content[:250].replace("\n", " ")
        citations.append(f"[{i}] From '{source}', page {page}:\n{excerpt}...")

    return f"🧠 Answer: {answer}\n\n📚 References:\n" + "\n\n".join(citations)

@tool
def get_weather_alerts(state: str, max_alerts: int = 5) -> str:
    """Get live alerts from the National Weather Service (US)."""
    try:
        url = f"https://api.weather.gov/alerts/active?area={state.upper()}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data.get("features"):
            return f"No active alerts in {state.upper()}."

        alerts = []
        for item in data["features"][:max_alerts]:
            props = item.get("properties", {})
            alerts.append(
                f"⚠️ {props.get('event')} in {props.get('areaDesc')}\n"
                f"{props.get('headline')}\nIssued: {props.get('sent')}\n"
                f"Description: {props.get('description')}\n"
                f"Instructions: {props.get('instruction')}\n"
            )
        return "\n\n".join(alerts)
    except Exception as e:
        return f"❌ Error fetching weather alerts: {str(e)}"

# Placeholder for image validation (to be integrated with real Gemini Vision model)
# @tool
# def validate_incident_with_image(description: str, image_url: str) -> str:
#     """Pretend to validate if an image matches a description (demo placeholder)."""
#     return f"🧪 Validated image at {image_url} for description: '{description}'"

# @tool
# def validate_incident_with_image(description: str, image_path: str) -> str:
#     """
#     Validate if an image matches the given description by sending the actual image
#     content (loaded from local path) and description to Gemini model.
#     """
#     try:
#         # Open the image from local path
#         image = PIL.Image.open(image_path)

#         # Send description and image to Gemini model
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=[description, image]
#         )

#         return response.text
#     except Exception as e:
#         return f"❌ Validation error: {str(e)}"

# ---------------------- AGENT INIT ----------------------
agent = initialize_agent(
    tools=[
        insert_incident_to_db,
        get_recent_incidents,
        answer_safety_faq,
        get_weather_alerts,
        # validate_incident_with_image
    ],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)


def validate_image(image_file, description=""):
    """
    image_file: A file-like object (e.g., from Streamlit uploader)
    Returns: string response from Gemini
    """
    # Handle memoryview from Streamlit
    if isinstance(image_file, memoryview):
        image_file = io.BytesIO(image_file.tobytes())

    # Read and encode the image
    image_bytes = image_file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Guess MIME type from PIL
    image = Image.open(io.BytesIO(image_bytes))
    mime_type = Image.MIME.get(image.format, "image/png")  # fallback to PNG

    # Build LangChain HumanMessage
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Is the image related to the description? Description: " + description},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded_image}"},
        ]
    )

    # Invoke Gemini
    result = llm.invoke([message])
    return result.content



# ---------------------- CLI LOOP ----------------------
if __name__ == "__main__":
    print("🛡️ Public Safety Chatbot (Indiana)")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            response = agent.run(user_input)
            print(f"\n🤖 Bot: {response}\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
