

!pip install -q gradio gtts pytz pillow

import gradio as gr
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, SystemMessage
import wikipedia
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from io import BytesIO
import pytz
from datetime import datetime
import base64
from PIL import Image



@tool
def scrape_url(url: str) -> str:
    """Scrape une URL."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content[:5000] if docs else "Erreur"
    except Exception as e:
        return f"Erreur: {str(e)}"

@tool
def search_wikipedia(query: str) -> str:
    """Wikipedia français."""
    try:
        wikipedia.set_lang("fr")
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Erreur: {str(e)}"

@tool
def search_web(query: str) -> str:
    """Recherche web."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={query}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        results = [r.get_text() for r in soup.find_all("a", class_="result__a")[:3]]
        return "\n".join(results) if results else "Aucun résultat"
    except Exception as e:
        return f"Erreur: {str(e)}"

@tool
def summarize_text(text: str) -> str:
    """Résume un texte."""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        response = llm.invoke(f"Résume:\n{text[:3000]}")
        return response.content
    except Exception as e:
        return f"Erreur: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Météo actuelle."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data["current_condition"][0]
            return f"""Météo à {city}:
Température: {current["temp_C"]}°C (ressenti {current["FeelsLikeC"]}°C)
Conditions: {current["weatherDesc"][0]["value"]}
Humidité: {current["humidity"]}%
Vent: {current["windspeedKmph"]} km/h"""
        return f"Erreur météo"
    except Exception as e:
        return f"Erreur: {str(e)}"

@tool
def get_current_time(timezone: str = "Europe/Paris") -> str:
    """Heure actuelle."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return f"""Heure actuelle ({timezone}):
Date: {now.strftime('%A %d %B %Y')}
Heure: {now.strftime('%H:%M:%S')}
Fuseau: {timezone}"""
    except:
        now = datetime.now()
        return f"Heure: {now.strftime('%H:%M:%S')}"

tools = [scrape_url, search_wikipedia, search_web, summarize_text, 
         get_weather, get_current_time]



llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

system_msg = SystemMessage(content="""Tu es un assistant IA avec 6 outils.
Utilise automatiquement le bon outil selon la question.""")

def agent_node(state):
    msgs = [system_msg] + state["messages"]
    return {"messages": [llm_with_tools.invoke(msgs)]}

def should_continue(state):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

print(" Agent créé avec 6 tools")

# ====================================================================
# FONCTION ANALYSE D'IMAGE AVEC VISION
# ====================================================================

def analyze_image_vision(image, description=""):
    """Analyse image avec GPT-4 Vision"""
    try:
        if image is None:
            return " Aucune image fournie"
        
        buffered = BytesIO()
        if isinstance(image, str):
            image = Image.open(image)
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        llm_vision = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=2000)
        
        prompt = f"""Tu es un expert en analyse d'images. Analyse cette image en détail.

Description/Contexte: {description if description else "Aucune"}

Fais une analyse complète et structurée:

###  1. Description Générale
- Type d'image (photo, illustration, document, etc.)
- Sujet principal
- Contexte et cadre

###  2. Observations Détaillées
- Éléments visuels présents
- Couleurs, formes, textures
- Composition et mise en page
- Texte visible (s'il y en a)
- Détails importants

###  3. Interprétation et Analyse
- Que représente cette image ?
- Message ou but apparent
- Points d'intérêt particuliers
- Qualité technique

###  4. Informations Extraites
- Texte lu (OCR si applicable)
- Données identifiables
- Éléments notables

###  5. Contexte et Recommandations
- Utilisation possible
- Suggestions d'amélioration
- Informations complémentaires utiles"""
        
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
        ])
        
        response = llm_vision.invoke([message])
        return response.content
        
    except Exception as e:
        return f" Erreur: {str(e)}\n\nVérifiez que GPT-4 Vision est activé sur votre compte OpenAI."

# ====================================================================
# FONCTION CHAT
# ====================================================================

def chat_function(message, history):
    try:
        result = app.invoke({"messages": [HumanMessage(content=message)]})
        answer = result["messages"][-1].content
        return answer
    except Exception as e:
        return f" Erreur: {str(e)}"

def generate_audio(text):
    """Génère l'audio pour un texte"""
    try:
        tts = gTTS(text=text[:500], lang='fr')
        temp_audio = "/tmp/response.mp3"
        tts.save(temp_audio)
        return temp_audio
    except:
        return None



custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-dark: #0f1419;
    --bg-secondary: #1a1f28;
    --bg-tertiary: #252b36;
    --accent-blue: #1d9bf0;
    --accent-purple: #7856ff;
    --text-primary: #e7e9ea;
    --text-secondary: #71767b;
    --border-color: #2f3336;
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

/* Header Simple */
.grok-header {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.grok-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-blue));
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.grok-header h1 {
    color: var(--text-primary) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

.grok-header p {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    margin: 0.3rem 0 0 0 !important;
}

/* Tabs */
.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: var(--text-secondary) !important;
    padding: 0.8rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* Chatbot */
.chat-container {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}

/* Input */
textarea, input[type="text"] {
    background: var(--bg-secondary) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    padding: 0.9rem !important;
    transition: all 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(29, 155, 240, 0.1) !important;
    outline: none !important;
}

/* Buttons */
.gradio-button {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1.3rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.gradio-button:hover {
    border-color: var(--accent-blue) !important;
    transform: translateY(-1px) !important;
}

.gradio-button-primary {
    background: var(--accent-blue) !important;
    border: none !important;
    color: white !important;
}

.gradio-button-primary:hover {
    background: #1a8cd8 !important;
    box-shadow: 0 8px 20px rgba(29, 155, 240, 0.3) !important;
}

/* Suggestions centrées */
.suggestions-center {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
}

.quick-suggestion {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: var(--text-primary);
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.9rem;
    font-weight: 500;
}

.quick-suggestion:hover {
    background: var(--accent-blue);
    border-color: var(--accent-blue);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(29, 155, 240, 0.3);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--bg-tertiary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-blue);
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 2rem;
    border-top: 1px solid var(--border-color);
}
"""

# ====================================================================
# INTERFACE GRADIO 
# ====================================================================

with gr.Blocks(css=custom_css, theme=gr.themes.Base(), title="🤖 Agent IA") as demo:
    

    last_user_msg = gr.State("")
    
    gr.HTML("""
    <div class="grok-header">
        <h1>🤖 Agent IA Multi-Fonctions</h1>
        <p>Chat avec l'agent intelligent</p>
    </div>
    """)
    
    with gr.Tabs():
        
        with gr.Tab("💬 Chat"):
            
            with gr.Column(visible=True, elem_classes=["suggestions-center"]) as suggestions_box:
                gr.HTML("<p style='text-align: center; color: var(--text-secondary); font-size: 1rem; margin-bottom: 1.5rem;'>💡 Suggestions rapides</p>")
                with gr.Row():
                    quick1 = gr.Button("🌤️ Météo Paris", elem_classes=["quick-suggestion"], size="sm")
                    quick2 = gr.Button("🕐 Heure actuelle", elem_classes=["quick-suggestion"], size="sm")
                with gr.Row():
                    quick3 = gr.Button("📚 Wiki Python", elem_classes=["quick-suggestion"], size="sm")
                    quick4 = gr.Button("🌐 Scrape Python.org", elem_classes=["quick-suggestion"], size="sm")
            
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                bubble_full_width=False,
                visible=False
            )
            
            with gr.Row(visible=False) as feedback_row:
                regenerate_btn = gr.Button("🔄", size="sm", scale=1, min_width=50)
                speak_btn = gr.Button("🔊", size="sm", scale=1, min_width=50)
                gr.HTML("<div style='flex-grow: 1;'></div>")
                clear_btn = gr.Button("🗑️ Effacer", size="sm", scale=1)
            
            audio_output = gr.Audio(visible=False, autoplay=True)
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Tapez votre message ici...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                submit_btn = gr.Button("📤", variant="primary", scale=1)
            
            
            def send_message(message, history, last_msg):
                if not message.strip():
                    return history, "", last_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                
                history.append((message, None))
                
                answer = chat_function(message, history)
                history[-1] = (message, answer)
                
                return (
                    history, 
                    "", 
                    message,
                    gr.update(visible=False),  
                    gr.update(visible=True),   
                    gr.update(visible=True)    
                )
            
            def regenerate_response(history, last_msg):
                if not history or not last_msg:
                    return history
                
                answer = chat_function(last_msg, history[:-1])
                history[-1] = (last_msg, answer)
                return history
            
            def speak_response(history):
                if not history:
                    return None
                
                last_answer = history[-1][1]
                if last_answer:
                    return generate_audio(last_answer)
                return None
            
            def clear_chat():
                return (
                    [], 
                    None, 
                    "",
                    gr.update(visible=True),   
                    gr.update(visible=False),  
                    gr.update(visible=False)   
                )
            
            # Events
            submit_btn.click(
                send_message,
                [msg, chatbot, last_user_msg],
                [chatbot, msg, last_user_msg, suggestions_box, chatbot, feedback_row]
            )
            
            msg.submit(
                send_message,
                [msg, chatbot, last_user_msg],
                [chatbot, msg, last_user_msg, suggestions_box, chatbot, feedback_row]
            )
            
            regenerate_btn.click(
                regenerate_response,
                [chatbot, last_user_msg],
                chatbot
            )
            
            speak_btn.click(
                speak_response,
                chatbot,
                audio_output
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot, audio_output, last_user_msg, suggestions_box, chatbot, feedback_row]
            )
            
            quick1.click(lambda: "Quel temps fait-il à Paris ?", outputs=msg)
            quick2.click(lambda: "Quelle heure est-il ?", outputs=msg)
            quick3.click(lambda: "Cherche sur Wikipedia: Python", outputs=msg)
            quick4.click(lambda: "Scrape https://www.python.org/ et résume", outputs=msg)
        
        with gr.Tab("🖼️ Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Votre Image",
                        type="pil",
                        height=400,
                        sources=["upload", "clipboard", "webcam"]
                    )
                    
                    image_desc = gr.Textbox(
                        label=" Contexte (optionnel)",
                        placeholder="Ajoutez du contexte...",
                        lines=3
                    )
                    
                    analyze_btn = gr.Button("🔬 Analyser", variant="primary")
                
                with gr.Column():
                    image_output = gr.Markdown(value="*Téléchargez une image pour commencer...*")
            
            analyze_btn.click(
                lambda img, desc: " Image requise" if img is None else analyze_image_vision(img, desc),
                inputs=[image_input, image_desc],
                outputs=image_output
            )
    
    gr.HTML("""
    <div class="footer">
        <p>Utiliser via API • Créé avec Gradio </p>
    </div>
    """)

print("="*70)
print(" LANCEMENT INTERFACE STYLE STREAMLIT")
print("="*70)

demo.launch(share=True, show_error=True)
