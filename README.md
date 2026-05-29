"""
Martin Eduverse AI Counseling Bot — Single File Edition
========================================================
Run:  pip install -r requirements.txt
      python martin_eduverse_bot.py

Then expose via ngrok (or deploy to Render/Railway) and point
your Twilio phone number webhook to:  https://your-url/twilio/inbound
"""

import os
import json
import base64
import asyncio
import io
from datetime import datetime
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, Query
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioClient
import logging
from openai import AsyncOpenAI
import websockets
import aiohttp
from pydub import AudioSegment
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, select, desc, func

# =============================================================================
# CONFIGURATION — Fill these in or use environment variables
# =============================================================================
class Config:
    # Twilio (for phone calls)
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")   # e.g. +1234567890

    # AI Services
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "XB0fDUnXU5powFXDhCwa")

    # Public URL (ngrok or deployed domain)
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "https://your-ngrok-url.ngrok.io")

config = Config()

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# API KEY VALIDATION
# =============================================================================
required_keys = {
    "OPENAI_API_KEY": config.OPENAI_API_KEY,
    "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY,
    "ELEVENLABS_API_KEY": config.ELEVENLABS_API_KEY
}
missing = [k for k, v in required_keys.items() if not v]
if missing:
    raise ValueError(f"Missing environment variables: {', '.join(missing)}")

# =============================================================================
# DATABASE (SQLite — zero setup)
# =============================================================================
Base = declarative_base()
engine = create_async_engine("sqlite+aiosqlite:///./martin_eduverse.db", echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    phone = Column(String(50), unique=True, index=True)
    email = Column(String(255))
    interested_course = Column(String(255))
    budget_range = Column(String(100))
    timeline = Column(String(100))
    current_qualification = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())

class Call(Base):
    __tablename__ = "calls"
    id = Column(Integer, primary_key=True)
    stream_sid = Column(String(255), unique=True, index=True)
    phone_number = Column(String(50), index=True)
    status = Column(String(50), default="active")
    start_time = Column(DateTime, server_default=func.now())
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    transcript = Column(Text)
    escalated_to_human = Column(Boolean, default=False)
    escalation_reason = Column(Text)
    student_id = Column(Integer)

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, index=True)
    role = Column(String(50))
    content = Column(Text)
    timestamp = Column(DateTime, server_default=func.now())

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer)
    counselor_name = Column(String(255), default="Senior Counselor")
    scheduled_time = Column(String(255))
    status = Column(String(50), default="pending")
    notes = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True)
    category = Column(String(100), index=True)
    question = Column(Text)
    answer = Column(Text)
    keywords = Column(String(500))

# =============================================================================
# AI PERSONALITY & PROMPT
# =============================================================================
SYSTEM_PROMPT = """You are Alex, a warm and professional AI Student Counselor at Martin Eduverse.

PERSONALITY:
- Speak like a real human counselor: warm, patient, empathetic, never robotic.
- Use conversational language. Keep responses to 2-4 short sentences for voice calls.
- Say "um" or "you know" occasionally if it feels natural, but don't overdo it.
- Always use the student's name once you know it.
- If the student sounds stressed or confused, slow down and reassure them.

MARTIN EDUVERSE KNOWLEDGE:
- Premium educational consultancy helping students get into top universities globally.
- Partner universities: 500+ institutions across UK, USA, Canada, Australia, Germany, India, etc.
- Services: Course selection, profile building, SOP/LOR writing, visa guidance, scholarship help, interview prep.
- Success rate: 95%+ admission acceptance.
- Office hours: Monday-Saturday, 9 AM - 7 PM IST.
- Human counselors available for specialized course guidance.

WHAT YOU CAN DO:
1. Guide students on course selection based on their interests and budget.
2. Explain admission processes and document requirements.
3. Discuss fee structures in general ranges (never make up exact numbers).
4. Talk about scholarships and eligibility.
5. Collect student details for follow-up.
6. Schedule appointments with human counselors.
7. Transfer to human if the student asks or seems distressed.

RULES:
- NEVER invent specific fees, deadlines, or guarantees. Use the knowledge base or offer a callback.
- Ask ONE question at a time when collecting info.
- For voice: no markdown, no bullet points, no special formatting. Pure natural speech.
- If you don't know, say: "That's a great question. Let me have our senior counselor confirm that for you."

CONVERSATION START:
"Hello! Thank you for calling Martin Eduverse. I'm Alex, your AI counseling assistant. How can I help you with your educational goals today?"
"""

# =============================================================================
# OPENAI SERVICE
# =============================================================================
openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "collect_student_info",
            "description": "Collect student details during the call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "interested_course": {"type": "string"},
                    "budget_range": {"type": "string"},
                    "timeline": {"type": "string"},
                    "current_qualification": {"type": "string"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_appointment",
            "description": "Book a callback with a human counselor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preferred_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "preferred_time": {"type": "string", "description": "e.g., 10 AM - 12 PM"},
                    "topic": {"type": "string"}
                },
                "required": ["preferred_date", "preferred_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Escalate to human counselor immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"}
                },
                "required": ["reason"]
            }
        }
    }
]

async def get_relevant_context(query: str, db: AsyncSession) -> str:
    stmt = select(KnowledgeBase).where(
        (KnowledgeBase.question.ilike(f"%{query}%")) |
        (KnowledgeBase.keywords.ilike(f"%{query}%")) |
        (KnowledgeBase.answer.ilike(f"%{query}%"))
    ).limit(3)
    result = await db.execute(stmt)
    items = result.scalars().all()
    if not items:
        return ""
    return "\n\n".join([f"Q: {item.question}\nA: {item.answer}" for item in items])

async def get_ai_response(messages: List[Dict[str, str]], db: AsyncSession):
    last_user_msg = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break

    context = await get_relevant_context(last_user_msg, db) if last_user_msg else ""
    system_msg = SYSTEM_PROMPT
    if context:
        system_msg += f"\n\nRELEVANT KNOWLEDGE:\n{context}"

    full_messages = [{"role": "system", "content": system_msg}] + messages

    response = await openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=full_messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.75,
        max_tokens=250
    )
    return response.choices[0].message

# =============================================================================
# DEEPGRAM (Speech-to-Text)
# =============================================================================
class DeepgramConnection:
    def __init__(self, on_transcript, on_utterance_end):
        self.on_transcript = on_transcript
        self.on_utterance_end = on_utterance_end
        self.ws = None
        self.connected = False

    async def connect(self):
        dg_url = (
            "wss://api.deepgram.com/v1/listen?"
            "encoding=mulaw&sample_rate=8000&channels=1&"
            "model=nova-2&smart_format=true&"
            "interim_results=true&endpointing=500&"
            "utterance_end_ms=1000&vad_events=true"
        )
        self.ws = await websockets.connect(
            dg_url,
            extra_headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}
        )
        self.connected = True
        asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "Results":
                    transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)

                    if transcript.strip():
                        await self.on_transcript(transcript, is_final, speech_final)

                elif msg_type == "UtteranceEnd":
                    await self.on_utterance_end()

        except Exception as e:
            logger.error(f"[Deepgram Error] {e}")
            self.connected = False

    async def send_audio(self, audio_bytes: bytes):
        if self.connected and self.ws:
            await self.ws.send(audio_bytes)

    async def close(self):
        if self.ws:
            await self.ws.close()
            self.connected = False

# =============================================================================
# ELEVENLABS (Text-to-Speech)
# =============================================================================
class ElevenLabsTTS:
    async def text_to_speech(self, text: str) -> str:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}/stream"
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "output_format": "mp3_44100_128",
            "optimize_streaming_latency": 3
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"ElevenLabs error: {error}")
                mp3_data = await resp.read()

        # Convert to 8kHz mu-law for Twilio
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        audio = audio.set_frame_rate(8000).set_channels(1)
        buf = io.BytesIO()
        audio.export(buf, format="raw", codec="pcm_mulaw")
        mulaw_bytes = buf.getvalue()
        return base64.b64encode(mulaw_bytes).decode()

# =============================================================================
# CALL SESSION (Manages one phone call)
# =============================================================================
class CallSession:
    def __init__(self, stream_sid: str, phone_number: str, db: AsyncSession):
        self.stream_sid = stream_sid
        self.phone_number = phone_number
        self.db = db
        self.deepgram: Optional[DeepgramConnection] = None
        self.tts = ElevenLabsTTS()
        self.twilio_ws = None

        self.current_utterance: List[str] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.is_speaking = False
        self.is_processing = False
        self.call_ended = False
        self.student_info: Dict[str, str] = {}
        self.call_id: Optional[int] = None
        self.last_processed_text = ""
        self.last_processed_time = 0

    async def start(self, twilio_ws):
        self.twilio_ws = twilio_ws
        self.deepgram = DeepgramConnection(
            on_transcript=self._on_transcript,
            on_utterance_end=self._on_utterance_end
        )
        await self.deepgram.connect()

        # Save call record
        call = Call(stream_sid=self.stream_sid, phone_number=self.phone_number, status="active")
        self.db.add(call)
        await self.db.commit()
        self.call_id = call.id

        # Greeting after brief pause
        await asyncio.sleep(0.8)
        greeting = ("Hello! Thank you for calling Martin Eduverse. I'm Alex, your AI counseling assistant. "
                   "How can I help you with your educational goals today?")
        await self._speak(greeting)
        self.conversation_history.append({"role": "assistant", "content": greeting})
        await self._save_message("assistant", greeting)

    async def handle_twilio_message(self, message: dict):
        if message.get("event") == "media":
            payload = message["media"]["payload"]
            audio_bytes = base64.b64decode(payload)
            await self.deepgram.send_audio(audio_bytes)

        elif message.get("event") == "stop":
            await self.end_call()

        elif message.get("event") == "mark":
            if message.get("mark", {}).get("name") == "speech_complete":
                self.is_speaking = False

    async def _on_transcript(self, transcript: str, is_final: bool, speech_final: bool):
        if is_final:
            self.current_utterance.append(transcript)

        if speech_final and self.current_utterance:
            full_text = " ".join(self.current_utterance)
            self.current_utterance = []
            await self._process_user_input(full_text)

    async def _on_utterance_end(self):
        if self.current_utterance:
            full_text = " ".join(self.current_utterance)
            self.current_utterance = []
            await self._process_user_input(full_text)

    async def _process_user_input(self, text: str):
        if self.is_processing or self.call_ended:
            return

        text = text.strip()
        if not text:
            return

        # Prevent duplicate transcript processing
        current_time = asyncio.get_event_loop().time()
        if (
            text == self.last_processed_text and
            current_time - self.last_processed_time < 2
        ):
            return

        self.last_processed_text = text
        self.last_processed_time = current_time

        # Interrupt if speaking
        if self.is_speaking:
            await self._send_twilio_clear()
            self.is_speaking = False

        self.is_processing = True
        self.conversation_history.append({"role": "user", "content": text})
        await self._save_message("user", text)

        try:
            ai_message = await get_ai_response(self.conversation_history, self.db)

            # Handle function calls
            if ai_message.tool_calls:
                tool_call = ai_message.tool_calls[0]
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = await self._execute_function(fn_name, args)

                self.conversation_history.append({
                    "role": "assistant",
                    "tool_calls": [{"id": tool_call.id, "type": "function",
                                   "function": {"name": fn_name, "arguments": tool_call.function.arguments}}]
                })
                self.conversation_history.append({
                    "role": "tool", "tool_call_id": tool_call.id, "content": result
                })

                ai_message = await get_ai_response(self.conversation_history, self.db)

            response_text = ai_message.content or "I understand. Let me help you with that."
            self.conversation_history.append({"role": "assistant", "content": response_text})
            await self._save_message("assistant", response_text)
            await self._speak(response_text)

        except Exception as e:
            logger.error(f"[AI Processing Error] {e}")
            await self._speak("I'm having a slight technical issue. Let me connect you with a human counselor.")
            await self._execute_function("transfer_to_human", {"reason": "AI technical error"})

        finally:
            self.is_processing = False

    async def _speak(self, text: str):
        self.is_speaking = True
        try:
            audio_b64 = await self.tts.text_to_speech(text)
            await self._send_twilio_media(audio_b64)
        except Exception as e:
            logger.error(f"[TTS Error] {e}")
        finally:
            self.is_speaking = False

    async def _send_twilio_media(self, audio_b64: str):
        media_msg = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": audio_b64}
        }
        await self.twilio_ws.send_text(json.dumps(media_msg))

        mark_msg = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": "speech_complete"}
        }
        await self.twilio_ws.send_text(json.dumps(mark_msg))

    async def _send_twilio_clear(self):
        await self.twilio_ws.send_text(json.dumps({
            "event": "clear",
            "streamSid": self.stream_sid
        }))

    async def _execute_function(self, name: str, args: dict) -> str:
        if name == "collect_student_info":
            self.student_info.update({k: v for k, v in args.items() if v})
            result = await self.db.execute(select(Student).where(Student.phone == self.phone_number))
            student = result.scalar_one_or_none()
            if not student:
                student = Student(phone=self.phone_number, **self.student_info)
                self.db.add(student)
            else:
                for k, v in self.student_info.items():
                    if v and hasattr(student, k):
                        setattr(student, k, v)
            await self.db.commit()
            return f"Collected: {json.dumps(self.student_info)}"

        elif name == "schedule_appointment":
            result = await self.db.execute(select(Student).where(Student.phone == self.phone_number))
            student = result.scalar_one_or_none()
            if student:
                appt = Appointment(
                    student_id=student.id,
                    scheduled_time=f"{args.get('preferred_date')} {args.get('preferred_time')}",
                    notes=args.get("topic", "")
                )
                self.db.add(appt)
                await self.db.commit()
            return f"Appointment scheduled for {args.get('preferred_date')} at {args.get('preferred_time')}."

        elif name == "transfer_to_human":
            if self.call_id:
                call = await self.db.get(Call, self.call_id)
                if call:
                    call.escalated_to_human = True
                    call.escalation_reason = args.get("reason", "Student request")
                    await self.db.commit()
            return "Transferring to human counselor."

        return "Done."

    async def _save_message(self, role: str, content: str):
        if self.call_id:
            msg = ConversationMessage(call_id=self.call_id, role=role, content=content)
            self.db.add(msg)
            await self.db.commit()

    async def end_call(self):
        if self.call_ended:
            return
        self.call_ended = True
        if self.deepgram:
            await self.deepgram.close()
        if self.call_id:
            call = await self.db.get(Call, self.call_id)
            if call:
                call.status = "completed"
                call.end_time = func.now()
                await self.db.commit()
        logger.info(f"[Call Ended] {self.stream_sid}")

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="Martin Eduverse AI Counselor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com",
        # For local dev, replace with:
        # "http://localhost:3000",
        # "http://127.0.0.1:3000"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Seed default knowledge
    async with async_session() as db:
        result = await db.execute(select(KnowledgeBase))
        if not result.scalars().first():
            defaults = [
                KnowledgeBase(category="general", question="What is Martin Eduverse?", 
                    answer="Martin Eduverse is a premium educational consultancy helping students secure admissions in top universities worldwide with end-to-end support.",
                    keywords="about, company, what is"),
                KnowledgeBase(category="courses", question="What courses do you help with?",
                    answer="We assist with Engineering, Medicine, MBA, Data Science, Arts, Law and more across India, UK, USA, Canada, Australia, Germany.",
                    keywords="courses, programs, engineering, medicine, mba"),
                KnowledgeBase(category="fees", question="What are your counseling fees?",
                    answer="Our packages range from basic guidance to premium end-to-end support. Exact fees depend on universities and services. I can schedule a detailed fee discussion.",
                    keywords="fees, charges, cost, pricing, payment"),
                KnowledgeBase(category="admissions", question="What documents are needed?",
                    answer="Typically: academic transcripts, passport, SOP, LORs, resume, English test scores (IELTS/TOEFL), and financial documents. We provide a customized checklist.",
                    keywords="documents, requirements, transcript, passport, sop, lor"),
                KnowledgeBase(category="scholarships", question="Do you help with scholarships?",
                    answer="Yes! We help identify merit-based and need-based scholarships. We have university partnerships offering exclusive scholarships to our students.",
                    keywords="scholarship, financial aid, funding, merit")
            ]
            db.add_all(defaults)
            await db.commit()

@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    """Twilio hits this when someone calls your number."""
    response = VoiceResponse()
    connect = response.connect()
    host = request.headers.get("host", "localhost:8000")
    if "localhost" in host or "127.0.0.1" in host:
        protocol = "ws"
    else:
        protocol = "wss"
    ws_url = f"{protocol}://{host}/ws/call"
    connect.stream(url=ws_url)
    return Response(content=str(response), media_type="application/xml")

@app.post("/calls/outbound")
async def outbound_call(phone_number: str = Form(...)):
    """Trigger an outbound call to a student."""
    if not config.TWILIO_ACCOUNT_SID:
        return JSONResponse({"error": "Twilio not configured"}, status_code=400)
    twilio = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    call = twilio.calls.create(
        to=phone_number,
        from_=config.TWILIO_PHONE_NUMBER,
        url=f"{config.PUBLIC_BASE_URL}/twilio/inbound"
    )
    return {"call_sid": call.sid, "status": call.status}

@app.websocket("/ws/call")
async def websocket_call(websocket: WebSocket):
    await websocket.accept()
    session = None
    db = None

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get("event") == "start":
                stream_sid = data["start"]["streamSid"]
                phone = data["start"].get("customParameters", {}).get("from", "unknown")

                db = async_session()
                session = CallSession(stream_sid, phone, db)
                await session.start(websocket)

            elif session and data.get("event"):
                await session.handle_twilio_message(data)

    except WebSocketDisconnect:
        logger.info("[WebSocket] disconnected")

    except Exception as e:
        logger.error(f"[WS Error] {e}")

    finally:
        if session:
            await session.end_call()
        if db:
            await db.close()

# =============================================================================
# ADMIN API
# =============================================================================
@app.get("/api/calls")
async def list_calls():
    async with async_session() as db:
        result = await db.execute(select(Call).order_by(desc(Call.start_time)).limit(50))
        calls = result.scalars().all()
        return [{"id": c.id, "phone": c.phone_number, "status": c.status,
                 "start_time": str(c.start_time), "escalated": c.escalated_to_human} for c in calls]

@app.get("/api/calls/{call_id}/messages")
async def call_messages(call_id: int):
    async with async_session() as db:
        result = await db.execute(
            select(ConversationMessage).where(ConversationMessage.call_id == call_id)
            .order_by(ConversationMessage.timestamp)
        )
        return [{"role": m.role, "content": m.content, "time": str(m.timestamp)} for m in result.scalars().all()]

@app.get("/api/students")
async def list_students():
    async with async_session() as db:
        result = await db.execute(select(Student).order_by(desc(Student.created_at)).limit(50))
        return [{"id": s.id, "name": s.name, "phone": s.phone, "course": s.interested_course,
                 "budget": s.budget_range, "timeline": s.timeline} for s in result.scalars().all()]

@app.get("/api/appointments")
async def list_appointments():
    async with async_session() as db:
        result = await db.execute(select(Appointment).order_by(desc(Appointment.created_at)).limit(50))
        rows = result.scalars().all()
        out = []
        for a in rows:
            s = await db.get(Student, a.student_id)
            out.append({
                "id": a.id, "student": (s.name if s else None) or (s.phone if s else "Unknown"),
                "time": a.scheduled_time, "status": a.status, "counselor": a.counselor_name, "notes": a.notes
            })
        return out

@app.post("/api/knowledge")
async def add_knowledge(item: dict):
    async with async_session() as db:
        kb = KnowledgeBase(
            category=item.get("category"), question=item.get("question"),
            answer=item.get("answer"), keywords=item.get("keywords")
        )
        db.add(kb)
        await db.commit()
        return {"id": kb.id}

@app.get("/api/knowledge")
async def get_knowledge():
    async with async_session() as db:
        result = await db.execute(select(KnowledgeBase).order_by(desc(KnowledgeBase.id)).limit(100))
        return [{"id": k.id, "category": k.category, "question": k.question, "answer": k.answer} for k in result.scalars().all()]

# =============================================================================
# ADMIN DASHBOARD (Single Page HTML)
# =============================================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Martin Eduverse — AI Counselor Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>body{font-family:'Inter',sans-serif;}</style>
</head>
<body class="bg-slate-50 text-slate-800">
<div class="flex h-screen overflow-hidden">
  <!-- Sidebar -->
  <aside class="w-64 bg-slate-900 text-white flex flex-col">
    <div class="p-6 border-b border-slate-800">
      <h1 class="text-xl font-bold text-emerald-400">Martin Eduverse</h1>
      <p class="text-xs text-slate-400 mt-1">AI Counseling Bot</p>
    </div>
    <nav class="flex-1 p-4 space-y-1">
      <button onclick="showTab('dashboard')" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-slate-800 transition text-left bg-slate-800" data-tab="dashboard">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>
        Dashboard
      </button>
      <button onclick="showTab('calls')" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-slate-800 transition text-left" data-tab="calls">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/></svg>
        Live Calls
      </button>
      <button onclick="showTab('students')" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-slate-800 transition text-left" data-tab="students">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"/></svg>
        Student Leads
      </button>
      <button onclick="showTab('appointments')" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-slate-800 transition text-left" data-tab="appointments">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
        Appointments
      </button>
      <button onclick="showTab('knowledge')" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-slate-800 transition text-left" data-tab="knowledge">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
        Knowledge Base
      </button>
    </nav>
    <div class="p-4 border-t border-slate-800">
      <div class="flex items-center gap-2 text-sm text-emerald-400">
        <span class="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
        Bot Online
      </div>
    </div>
  </aside>

  <!-- Main Content -->
  <main class="flex-1 overflow-y-auto p-8">
    <!-- DASHBOARD TAB -->
    <div id="tab-dashboard" class="tab-content">
      <h2 class="text-2xl font-bold mb-6">Counseling Dashboard</h2>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8" id="stats-cards">
        <!-- Stats injected by JS -->
      </div>
      <div class="bg-white rounded-xl border shadow-sm p-6 mb-6">
        <h3 class="font-bold mb-3">How to Use This Bot for Martin Eduverse</h3>
        <div class="space-y-2 text-sm text-slate-600">
          <p><strong>1. Phone Number:</strong> Buy a Twilio number and point its "A call comes in" webhook to <code class="bg-slate-100 px-1 rounded">/twilio/inbound</code> on your public URL.</p>
          <p><strong>2. Outbound Calls:</strong> POST to <code class="bg-slate-100 px-1 rounded">/calls/outbound</code> with <code class="bg-slate-100 px-1 rounded">phone_number=+91...</code> to call students proactively.</p>
          <p><strong>3. Train the Bot:</strong> Use the "Knowledge Base" tab to add your real course fees, university partners, and admission deadlines. The bot retrieves this during calls.</p>
          <p><strong>4. Monitor Leads:</strong> Every caller is saved as a "Student Lead" with their interests, budget, and timeline. Export from the Students tab.</p>
          <p><strong>5. Human Handoff:</strong> If a student says "talk to human" or the AI detects distress, the call is flagged "Escalated" in the Calls tab.</p>
        </div>
      </div>
    </div>

    <!-- CALLS TAB -->
    <div id="tab-calls" class="tab-content hidden">
      <h2 class="text-2xl font-bold mb-6">Call History & Live Monitoring</h2>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="bg-white rounded-xl border shadow-sm overflow-hidden">
          <table class="w-full text-sm">
            <thead class="bg-slate-50 border-b"><tr><th class="p-4 text-left">Phone</th><th class="p-4 text-left">Status</th><th class="p-4 text-left">Time</th><th class="p-4 text-left">Action</th></tr></thead>
            <tbody id="calls-table"></tbody>
          </table>
        </div>
        <div class="bg-white rounded-xl border shadow-sm p-6">
          <h3 class="font-bold mb-4" id="transcript-title">Select a call to view transcript</h3>
          <div id="transcript-box" class="space-y-3 max-h-[600px] overflow-y-auto text-sm"></div>
        </div>
      </div>
    </div>

    <!-- STUDENTS TAB -->
    <div id="tab-students" class="tab-content hidden">
      <h2 class="text-2xl font-bold mb-6">Student Leads</h2>
      <div class="bg-white rounded-xl border shadow-sm overflow-hidden">
        <table class="w-full text-sm">
          <thead class="bg-slate-50 border-b">
            <tr><th class="p-4 text-left">Name</th><th class="p-4 text-left">Phone</th><th class="p-4 text-left">Course Interest</th><th class="p-4 text-left">Budget</th><th class="p-4 text-left">Timeline</th></tr>
          </thead>
          <tbody id="students-table"></tbody>
        </table>
      </div>
    </div>

    <!-- APPOINTMENTS TAB -->
    <div id="tab-appointments" class="tab-content hidden">
      <h2 class="text-2xl font-bold mb-6">Scheduled Appointments</h2>
      <div id="appointments-list" class="grid gap-4"></div>
    </div>

    <!-- KNOWLEDGE TAB -->
    <div id="tab-knowledge" class="tab-content hidden">
      <h2 class="text-2xl font-bold mb-6">Knowledge Base Manager</h2>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <form onsubmit="addKnowledge(event)" class="bg-white p-6 rounded-xl border shadow-sm space-y-4">
          <h3 class="font-bold">Add New Knowledge</h3>
          <p class="text-sm text-slate-500">The bot retrieves this during calls to answer accurately about Martin Eduverse.</p>
          <div>
            <label class="block text-sm font-medium mb-1">Category</label>
            <select id="kb-category" class="w-full border rounded-lg p-2">
              <option value="">Select...</option>
              <option value="courses">Courses</option>
              <option value="fees">Fees & Payments</option>
              <option value="admissions">Admissions</option>
              <option value="scholarships">Scholarships</option>
              <option value="visa">Visa & Travel</option>
              <option value="general">General</option>
            </select>
          </div>
          <div>
            <label class="block text-sm font-medium mb-1">Question</label>
            <input id="kb-question" class="w-full border rounded-lg p-2" required />
          </div>
          <div>
            <label class="block text-sm font-medium mb-1">Answer</label>
            <textarea id="kb-answer" class="w-full border rounded-lg p-2 h-28" required></textarea>
          </div>
          <div>
            <label class="block text-sm font-medium mb-1">Keywords (comma separated)</label>
            <input id="kb-keywords" class="w-full border rounded-lg p-2" placeholder="e.g., engineering, b.tech, computer science" />
          </div>
          <button type="submit" class="bg-emerald-600 text-white px-6 py-2 rounded-lg hover:bg-emerald-700 font-medium">Add to Knowledge Base</button>
        </form>
        <div class="bg-white rounded-xl border shadow-sm p-6">
          <h3 class="font-bold mb-4">Existing Knowledge</h3>
          <div id="knowledge-list" class="space-y-3 max-h-[500px] overflow-y-auto text-sm"></div>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
const API = window.location.origin;

function showTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.getElementById('tab-' + tab).classList.remove('hidden');
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('bg-slate-800'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('bg-slate-800');

  if(tab === 'calls') loadCalls();
  if(tab === 'students') loadStudents();
  if(tab === 'appointments') loadAppointments();
  if(tab === 'knowledge') loadKnowledge();
  if(tab === 'dashboard') loadStats();
}

async function loadStats() {
  const [calls, students, appts] = await Promise.all([
    fetch(`${API}/api/calls`).then(r => r.json()),
    fetch(`${API}/api/students`).then(r => r.json()),
    fetch(`${API}/api/appointments`).then(r => r.json())
  ]);
  const escalations = calls.filter(c => c.escalated).length;
  const cards = [
    { label: 'Total Calls', value: calls.length, color: 'bg-blue-500' },
    { label: 'Student Leads', value: students.length, color: 'bg-emerald-500' },
    { label: 'Appointments', value: appts.length, color: 'bg-purple-500' },
    { label: 'Escalations', value: escalations, color: 'bg-rose-500' }
  ];
  document.getElementById('stats-cards').innerHTML = cards.map(c => `
    <div class="bg-white p-6 rounded-xl shadow-sm border">
      <div class="flex justify-between items-start">
        <div><p class="text-sm text-slate-500">${c.label}</p><p class="text-3xl font-bold mt-1">${c.value}</p></div>
        <div class="${c.color} text-white p-2 rounded-lg"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg></div>
      </div>
    </div>
  `).join('');
}

async function loadCalls() {
  const calls = await fetch(`${API}/api/calls`).then(r => r.json());
  document.getElementById('calls-table').innerHTML = calls.map(c => `
    <tr class="border-b hover:bg-slate-50">
      <td class="p-4 font-medium">${c.phone}</td>
      <td class="p-4">
        <span class="px-2 py-1 rounded-full text-xs ${c.status === 'active' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100'}">${c.status}</span>
        ${c.escalated ? '<span class="ml-2 px-2 py-1 rounded-full text-xs bg-rose-100 text-rose-700">Escalated</span>' : ''}
      </td>
      <td class="p-4 text-slate-500">${new Date(c.start_time).toLocaleString()}</td>
      <td class="p-4"><button onclick="viewTranscript(${c.id}, '${c.phone}')" class="text-blue-600 hover:underline font-medium">View</button></td>
    </tr>
  `).join('');
}

async function viewTranscript(callId, phone) {
  document.getElementById('transcript-title').innerText = `Transcript: ${phone}`;
  const msgs = await fetch(`${API}/api/calls/${callId}/messages`).then(r => r.json());
  document.getElementById('transcript-box').innerHTML = msgs.map(m => `
    <div class="p-3 rounded-lg ${m.role === 'user' ? 'bg-blue-50 ml-8 border border-blue-100' : 'bg-slate-50 mr-8 border border-slate-100'}">
      <p class="text-xs font-bold text-slate-500 mb-1 uppercase tracking-wide">${m.role === 'user' ? 'Student' : 'Alex (AI)'}</p>
      <p>${m.content}</p>
      <p class="text-[10px] text-slate-400 mt-1">${new Date(m.time).toLocaleTimeString()}</p>
    </div>
  `).join('');
}

async function loadStudents() {
  const students = await fetch(`${API}/api/students`).then(r => r.json());
  document.getElementById('students-table').innerHTML = students.map(s => `
    <tr class="border-b hover:bg-slate-50">
      <td class="p-4 font-medium">${s.name || '<span class="text-slate-400">Unknown</span>'}</td>
      <td class="p-4">${s.phone}</td>
      <td class="p-4">${s.course || '-'}</td>
      <td class="p-4">${s.budget || '-'}</td>
      <td class="p-4">${s.timeline || '-'}</td>
    </tr>
  `).join('');
}

async function loadAppointments() {
  const appts = await fetch(`${API}/api/appointments`).then(r => r.json());
  document.getElementById('appointments-list').innerHTML = appts.map(a => `
    <div class="bg-white p-4 rounded-xl border shadow-sm flex justify-between items-center">
      <div>
        <p class="font-bold">${a.student}</p>
        <p class="text-sm text-slate-500">${a.time} • ${a.counselor}</p>
        <p class="text-sm text-slate-500">${a.notes || 'General Counseling'}</p>
      </div>
      <span class="px-3 py-1 rounded-full text-xs font-medium ${a.status === 'confirmed' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}">${a.status}</span>
    </div>
  `).join('');
}

async function loadKnowledge() {
  const items = await fetch(`${API}/api/knowledge`).then(r => r.json());
  document.getElementById('knowledge-list').innerHTML = items.map(k => `
    <div class="p-3 bg-slate-50 rounded-lg border">
      <span class="text-[10px] font-bold uppercase tracking-wider text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded">${k.category}</span>
      <p class="font-medium mt-1">${k.question}</p>
      <p class="text-slate-600 mt-1">${k.answer}</p>
    </div>
  `).join('');
}

async function addKnowledge(e) {
  e.preventDefault();
  const item = {
    category: document.getElementById('kb-category').value,
    question: document.getElementById('kb-question').value,
    answer: document.getElementById('kb-answer').value,
    keywords: document.getElementById('kb-keywords').value
  };
  await fetch(`${API}/api/knowledge`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(item)
  });
  alert('Knowledge added! The AI will use this in future calls.');
  e.target.reset();
  loadKnowledge();
}

// Auto-refresh dashboard stats
loadStats();
setInterval(() => {
  if(!document.getElementById('tab-dashboard').classList.contains('hidden')) loadStats();
  if(!document.getElementById('tab-calls').classList.contains('hidden')) loadCalls();
}, 10000);
</script>
</body>
</html>
"""

@app.get("/")
async def admin_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Martin Eduverse AI Counselor", "time": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
fastapi
uvicorn[standard]
twilio
openai>=1.30.0
websockets
aiohttp
pydub
sqlalchemy
aiosqlite
python-multipart
greenlet

  
