import io, uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from faiss_client import query_similar, add_invoice
from voice_service import stt_streaming, tts_stream
import openai
import uvicorn
import wave
import numpy as np
from PIL import Image
import pytesseract, pdf2image
from sentence_transformers import SentenceTransformer


openai.api_key = "USE YOUR API KEY"
app = FastAPI()
MAX_CONFIDENCE_UNSURE = 0.4

def wav_bytes_to_pcm(audio_bytes):
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        pcm_data = np.frombuffer(frames, dtype=np.int16)
    return pcm_data

@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    pcm_audio = wav_bytes_to_pcm(audio_bytes)

    text = await stt_streaming(pcm_audio)

    
    docs = query_similar(text, embed_fn=stt_streaming)  
    context = "\n\n".join(docs['documents'][0])
    messages = [
        {"role": "system", "content": "You are an invoice assistant."},
        {"role": "user", "content": f"Use this context:\n{context}\n\nUser: {text}"}
    ]

    
    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=False)
    answer = resp.choices[0].message.content

    
    if "I don't know" in answer.lower():
        audio_resp = tts_stream("I'm not certain, let me connect you to a live agent.")
        return StreamingResponse(io.BytesIO(audio_resp), media_type="audio/mpeg")

    
    audio_resp = tts_stream(answer)
    return StreamingResponse(io.BytesIO(audio_resp), media_type="audio/mpeg")


embedder = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/upload")
async def upload_invoices(files: list[UploadFile] = File(...)):
    uploaded = []
    for f in files:
        content = await f.read()
        images = (
            pdf2image.convert_from_bytes(content) 
            if f.filename.lower().endswith(".pdf") 
            else [Image.open(io.BytesIO(content))]
        )
        text = "\n".join(pytesseract.image_to_string(img) for img in images)
        inv_id = str(uuid.uuid4())[:8]
        metadata = {"invoice_id": inv_id, "filename": f.filename}
        add_invoice(inv_id, text, metadata, embedder.encode)
        uploaded.append({"invoice_id": inv_id})
    return {"uploaded": uploaded}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
