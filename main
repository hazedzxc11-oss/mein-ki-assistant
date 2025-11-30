# app.py
"""
Single-file Modern AI Assistant (OpenAI v1.x + FastAPI)
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
import requests
from functools import wraps

# NEUE IMPORTS FÜR PDF-FUNKTION
from io import BytesIO
import pypdf 

# --- load environment ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o") 
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")
TRELLO_KEY = os.getenv("TRELLO_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
TRELLO_LIST_ID = os.getenv("TRELLO_LIST_ID")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_API_HOST = os.getenv("STABILITY_API_HOST", "https://api.stability.ai")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY muss in der .env gesetzt sein.")

# --- Modern Client Initialization ---
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-assistant")

app = FastAPI(title="Modern AI Assistant")

# [Restlicher Code für rate_limit und require_service_key bleibt unverändert]

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30
_rate_store = {}

def rate_limit(key: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    arr = _rate_store.get(key, [])
    arr = [t for t in arr if t > window_start]
    if len(arr) >= RATE_LIMIT_MAX:
        _rate_store[key] = arr
        return False
    arr.append(now)
    _rate_store[key] = arr
    return True

def require_service_key(func):
    @wraps(func)
    async def wrapper(*args, x_api_key: Optional[str] = Header(None), request: Request = None, **kwargs):
        if SERVICE_API_KEY:
            if x_api_key != SERVICE_API_KEY:
                raise HTTPException(status_code=401, detail="Ungültiger Service API Key.")
        key = x_api_key if x_api_key else (request.client.host if request else "anon")
        if not rate_limit(key):
            raise HTTPException(status_code=429, detail="Zu viele Anfragen.")
        return await func(*args, **kwargs)
    return wrapper

# --- Pydantic model ---
class Prompt(BaseModel):
    text: str
    mode: Optional[str] = "chat"
    notify_apps: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None

# --- Helpers ---
def build_system_prompt(mode: str):
    if mode == "code":
        return "Du bist ein Senior KI-Entwickler. Antworte nur mit Code und kurzen Kommentaren. Nutze Best Practices."
    if mode == "design":
        return "Du bist ein UI/UX-Experte. Erstelle Layout-Konzepte, CSS-Vorschläge und Bild-Prompts."
    if mode == "both":
        return "Kombiniere Code-Implementierung mit Design-Ästhetik."
    return "Du bist ein hilfreicher Assistent."

# --- Modern OpenAI Wrapper (Async) ---
async def ai_generate(prompt: str, mode: str = "chat", max_tokens: int = 800, temperature: float = 0.7) -> str:
    system_prompt = build_system_prompt(mode)
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.exception("OpenAI error")
        raise HTTPException(status_code=500, detail=f"OpenAI-Fehler: {str(e)}")

# --- PDF Helper (NEU) ---
def extract_text_from_pdf(file: UploadFile) -> str:
    """Extrahiert Text aus einem hochgeladenen PDF-Objekt."""
    try:
        file_content = file.file.read()
        pdf_file = BytesIO(file_content)
        
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Begrenze den Text auf 8000 Zeichen, um Token-Limits zu vermeiden
        return text[:8000]
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren von PDF: {e}")
        return f"FEHLER: Konnte PDF nicht verarbeiten. {e}"


# --- App Integrations ---
def create_trello_card(name: str, desc: str):
    if not (TRELLO_KEY and TRELLO_TOKEN and TRELLO_LIST_ID):
        return {"error": "Trello config missing"}
    url = "https://api.trello.com/1/cards"
    params = {"key": TRELLO_KEY, "token": TRELLO_TOKEN, "idList": TRELLO_LIST_ID, "name": name, "desc": desc}
    try:
        r = requests.post(url, params=params, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def send_slack_message(message: str):
    if not SLACK_WEBHOOK:
        return {"error": "Slack config missing"}
    try:
        requests.post(SLACK_WEBHOOK, json={"text": message}, timeout=5)
        return {"status": "sent"}
    except Exception as e:
        return {"error": str(e)}

def generate_image_stability(prompt: str):
    if not STABILITY_API_KEY:
        return {"error": "STABILITY_API_KEY missing"}
    url = f"{STABILITY_API_HOST}/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"text_prompts": [{"text": prompt}], "cfg_scale": 7, "height": 1024, "width": 1024, "samples": 1}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            return {"error": r.text}
        data = r.json()
        return {"data": [{"b64_json": data["artifacts"][0]["base64"]}]}
    except Exception as e:
        return {"error": str(e)}

# --- Frontend (HTML) - Inklusive PDF-Upload und Mobile-Optimierung ---
INDEX_HTML = r"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
  <title>Mein KI Modell (v1.x)</title>
  <style>
    :root { 
        --primary: #0f172a; 
        --accent: #3b82f6; 
        --bg: #f8fafc; 
        --text-color: #334155;
        --border-color: #cbd5e1;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: var(--bg);
        color: var(--text-color);
        margin: 0;
        padding: 15px; /* Reduzierter Rand für Handy-Screens */
        display: flex;
        justify-content: center;
    }
    .container {
        width: 100%;
        max-width: 800px;
        background: #fff;
        padding: 20px; /* Reduzierter Innenabstand */
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    h1 {
        color: var(--primary);
        margin-top: 0;
        font-size: 1.4rem; /* Etwas kleiner für Mobile */
    }
    .control-group { margin-bottom: 12px; }
    label { display: block; margin-bottom: 5px; font-weight: 600; font-size: 0.9rem; }
    textarea, select, input[type=text], input[type=password], input[type=file] {
        width: 100%;
        padding: 10px;
        border: 1px solid var(--border-color);
        border-radius: 6px; 
        font-family: inherit;
        box-sizing: border-box;
        resize: vertical;
        font-size: 1rem;
    }
    .options {
        display: flex;
        flex-direction: column; /* Optionen untereinander auf Mobile */
        gap: 10px; 
        margin: 15px 0;
    }
    button {
        background: var(--accent);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: opacity 0.2s;
    }
    button:hover:not(:disabled) { opacity: 0.9; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    pre {
        background: #1e293b;
        color: #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
        white-space: pre-wrap;
        font-size: 0.85rem; 
    }
    #imagewrap img {
        max-width: 100%;
        border-radius: 8px;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status { font-size: 0.8rem; color: #64748b; margin-top: 5px; text-align: right; }
    /* Neue Klasse für den PDF-Bereich */
    .pdf-mode { border-left: 3px solid #f97316; padding-left: 10px; margin-top: 20px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🚀 Mein KI Assistant</h1>
    
    <div class="control-group">
        <label>Service Key</label>
        <input type="password" id="apikey" placeholder="SERVICE_API_KEY eingeben...">
    </div>

    <div class="control-group">
        <label>Modus</label>
        <select id="mode">
            <option value="chat">💬 Chat</option>
            <option value="code">💻 Code Expert</option>
            <option value="design">🎨 UI/Design</option>
            <option value="both">🚀 Full Stack</option>
            <option value="pdf">📄 PDF-Zusammenfassung</option> </select>
    </div>

    <div class="control-group" id="prompt-group">
        <label>Prompt / Befehl</label>
        <textarea id="prompt" placeholder="Was möchtest du erschaffen?"></textarea>
    </div>
    
    <div class="control-group pdf-mode" id="file-group" style="display:none;">
        <label>PDF-Datei</label>
        <input type="file" id="pdf-file" accept=".pdf">
        <small style="color: #f97316;">(Ignoriert den Prompt. Die KI fasst das PDF zusammen)</small>
    </div>


    <div class="options">
        <label style="font-weight:normal;display:flex;align-items:center;gap:8px;">
            <input type="checkbox" id="notify"/> Apps benachrichtigen (Slack/Trello)
        </label>
        <label style="font-weight:normal;display:flex;align-items:center;gap:8px;">
            <input type="checkbox" id="autoimg"/> Bild generieren (Stability)
        </label>
    </div>

    <button id="send">Generieren</button>
    <div id="status" class="status">Bereit</div>

    <div id="imagewrap"></div>
    <h3>Antwort:</h3>
    <pre id="result">Noch keine Daten.</pre>
  </div>

  <script>
    const btn = document.getElementById("send");
    const statusDiv = document.getElementById("status");
    const modeSelect = document.getElementById("mode");
    const promptGroup = document.getElementById("prompt-group");
    const fileGroup = document.getElementById("file-group");

    // JS-Logik für PDF-Modus (zeigt Dateifeld oder Prompt-Feld an)
    modeSelect.onchange = () => {
        if (modeSelect.value === 'pdf') {
            promptGroup.style.display = 'none';
            fileGroup.style.display = 'block';
        } else {
            promptGroup.style.display = 'block';
            fileGroup.style.display = 'none';
        }
    };
    
    btn.onclick = async () => {
      const mode = modeSelect.value;
      const prompt = document.getElementById("prompt").value;
      const fileInput = document.getElementById("pdf-file");
      const notify = document.getElementById("notify").checked;
      const autoimg = document.getElementById("autoimg").checked;
      const apikey = document.getElementById("apikey").value;
      
      const isPdfMode = mode === 'pdf';

      if (!isPdfMode && !prompt) return alert("Bitte Prompt eingeben!");
      if (isPdfMode && (!fileInput.files || fileInput.files.length === 0)) return alert("Bitte eine PDF-Datei auswählen!");

      btn.disabled = true;
      btn.textContent = "KI denkt nach...";
      statusDiv.textContent = "Sende Anfrage an Server...";
      document.getElementById("result").textContent = "";
      document.getElementById("imagewrap").innerHTML = "";
      
      let endpoint = isPdfMode ? "/api/pdf_summary" : "/api/" + (mode === "chat" ? "chat" : mode);
      let headers = { "X-API-Key": apikey };
      let body;

      if (isPdfMode) {
          // Dateiupload über FormData
          body = new FormData();
          body.append("file", fileInput.files[0]);
          // multipart/form-data braucht keinen Content-Type Header
      } else {
          // JSON-Body für Text-Prompts
          headers["Content-Type"] = "application/json";
          body = JSON.stringify({ text: prompt, mode: mode, notify_apps: notify, metadata: { generate_image: autoimg } });
      }

      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: headers,
          body: body
        });
        
        if(res.status === 401) throw new Error("API Key falsch (401)");
        if(res.status === 429) throw new Error("Zu viele Anfragen (429)");
        
        const j = await res.json();
        btn.textContent = "Fertig!";
        statusDiv.textContent = "Antwort empfangen.";
        
        let textOutput;
        if (isPdfMode) {
            textOutput = `Zusammenfassung für ${j.filename}:\n\n` + (j.summary || JSON.stringify(j, null, 2));
        } else {
            textOutput = j.output || j.code || j.design || j.both || JSON.stringify(j, null, 2);
        }
        document.getElementById("result").textContent = textOutput;

        // Image Handling
        if (j.image && j.image.data && j.image.data[0]) {
            const b64 = j.image.data[0].b64_json;
            const img = new Image();
            img.src = "data:image/png;base64," + b64;
            document.getElementById("imagewrap").appendChild(img);
        } else if (j.image && j.image.error) {
            statusDiv.textContent += " (Bild-Fehler: " + j.image.error + ")";
        }

      } catch (e) {
        document.getElementById("result").textContent = "Fehler: " + e.message;
        statusDiv.textContent = "Ein Fehler ist aufgetreten.";
      } finally {
        btn.disabled = false;
        btn.textContent = "Generieren";
      }
    };
  </script>
</body>
</html>
"""

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

@app.post("/api/chat")
# ... (Funktion wie gehabt)

@app.post("/api/code")
# ... (Funktion wie gehabt)

@app.post("/api/design")
# ... (Funktion wie gehabt)

@app.post("/api/both")
@require_service_key
async def api_both(p: Prompt):
    logger.info("Full Stack request")
    # 🌟 KORRIGIERTER CODE
    out = await ai_generate(p.text, mode="both", max_tokens=2500, temperature=0.4)
    
    image_result = None
    if p.metadata and p.metadata.get("generate_image"):
        image_result = generate_image_stability(f"UI Design screenshot for: {p.text}")

    return {"both": out, "image": image_result}

# --- NEUER ENDPUNKT FÜR PDF ---
@app.post("/api/pdf_summary")
@require_service_key
async def api_pdf_summary(file: UploadFile = File(...)):
    logger.info("PDF Summary request")

    # 1. Text aus PDF extrahieren
    pdf_text = extract_text_from_pdf(file)
    
    if pdf_text.startswith("FEHLER"):
        raise HTTPException(status_code=500, detail=pdf_text)
        
    # 2. Prompt für die KI erstellen
    summary_prompt = (
        "Du bist ein Experte für Dokumentenzusammenfassungen. "
        "Bitte fasse den folgenden Text aus einer PDF-Datei prägnant und strukturiert in Stichpunkten zusammen. "
        "Originaltext:\n\n" + pdf_text
    )

    # 3. KI zur Generierung aufrufen
    out = await ai_generate(
        summary_prompt, 
        mode="chat", 
        max_tokens=1000, 
        temperature=0.0
    )

    return {"filename": file.filename, "summary": out, "char_count": len(pdf_text)}
