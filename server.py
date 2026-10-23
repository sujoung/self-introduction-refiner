"""Swedish self-introduction game: local Pianissimo transcription + Gemini scoring.

Run: hrun uv run server.py  ->  http://localhost:8000
"""

import io
import json
import re
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy
import onnx_asr
from google import genai

PORT = 8000
PUBLIC = Path(__file__).parent / "public"

GEMINI_MODEL = "gemini-3.8-flash"
FILLERS = ("eh", "ehm", "öh", "hm", "liksom", "typ", "alltså", "asså", "ba")

RUBRIC = """You are a strict but encouraging pitch coach. The user had 60 seconds to introduce
themselves in Swedish. Score the transcript below against the rubric. For every criterion,
quote the exact words from the transcript as evidence, or write "missing". Never invent content.

Rubric (points 0-2 each):
- name_and_role: states their name and current role or studies
- background: mentions 1-2 relevant past experiences
- concrete_proof: at least one specific example with a number, project, or outcome
- ask_or_goal: says what they want next (job, collaboration, next step)
- structure: clear opening -> body -> close, no looping back
- clarity: few filler words (eh, liksom, typ, alltså); 0 points if more than 5 fillers
- length: 120-180 words is ideal for 60 seconds; 1 point if within 40 words of that range, else 0
{custom}
Write feedback as 3-5 short, concrete, actionable bullets in English.
Write restructured_intro as an improved version of THE USER'S OWN intro in Swedish, 120-180 words,
keeping their facts, in the order: who I am, what I have done (with proof), what I want.
List filler_words you found with counts (empty list if none).

Transcript:
<transcript>
{transcript}
</transcript>
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "array", "items": {"type": "object", "properties": {
            "criterion": {"type": "string"}, "points": {"type": "integer"},
            "max": {"type": "integer"}, "evidence": {"type": "string"}},
            "required": ["criterion", "points", "max", "evidence"]}},
        "feedback": {"type": "array", "items": {"type": "string"}},
        "restructured_intro": {"type": "string"},
        "filler_words": {"type": "array", "items": {"type": "object", "properties": {
            "word": {"type": "string"}, "count": {"type": "integer"}}, "required": ["word", "count"]}},
    },
    "required": ["scores", "feedback", "restructured_intro", "filler_words"],
}


def word_stats(text):
    words = re.findall(r"\w+", text.lower())
    fillers = {filler: words.count(filler) for filler in FILLERS if filler in words}
    return {"word_count": len(words), "filler_count": sum(fillers.values()), "fillers": fillers}


def score(transcript, notes=""):
    custom = ""
    if notes.strip():
        custom = ("- custom (0-2 each): the user asked for these points, score each one as its own "
                  "criterion named custom:<short label>:\n" + notes.strip() + "\n")
    started = time.monotonic()
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=RUBRIC.format(custom=custom, transcript=transcript),
        config={"response_mime_type": "application/json", "response_schema": SCHEMA},
    )
    print(f"scored in {time.monotonic() - started:.1f}s", file=sys.stderr)
    result = json.loads(response.text)
    result["total"] = sum(item["points"] for item in result["scores"])
    result["max_total"] = sum(item["max"] for item in result["scores"])
    result.update(word_stats(transcript))
    return result


def transcribe(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        if (rate, channels, width) != (16000, 1, 2):
            raise ValueError(f"expected mono 16 kHz 16-bit WAV, got {channels}ch {rate} Hz {width * 8}-bit")
        samples = numpy.frombuffer(wav.readframes(wav.getnframes()), numpy.int16).astype(numpy.float32) / 32768
    started = time.monotonic()
    text = model.recognize(samples, sample_rate=rate)
    print(f"transcribed {len(samples) / rate:.1f}s of audio in {time.monotonic() - started:.2f}s", file=sys.stderr)
    return text


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply(200, (PUBLIC / "index.html").read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        try:
            if self.path == "/transcribe":
                result = {"text": transcribe(body)}
            elif self.path == "/score":
                request = json.loads(body)
                result = score(request["transcript"], request.get("notes", ""))
            else:
                self.reply(404, b"not found", "text/plain")
                return
            self.reply(200, json.dumps(result, ensure_ascii=False).encode(), "application/json")
        except Exception as error:
            print(f"error: {error!r}", file=sys.stderr)
            self.reply(500, json.dumps({"error": f"{type(error).__name__}: {error}"}).encode(), "application/json")

    def reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    started = time.monotonic()
    model = onnx_asr.load_model("KlangAI/pianissimo-sv-onnx", quantization="int8")
    print(f"model loaded in {time.monotonic() - started:.1f}s", file=sys.stderr)
    gemini = genai.Client()
    print(f"listening on http://localhost:{PORT}", file=sys.stderr)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
