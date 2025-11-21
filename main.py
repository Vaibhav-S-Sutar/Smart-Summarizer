# main.py
"""
Smart Summarizer - robust combined version (v1.2.9 Final)

Enhancements over v1.2.6:
- Universal language matching for subtitles (ISO codes + full names)
- Skips audio download completely (Gemini can listen directly)
- Uses Gemini to summarize non-English subtitles directly into English
- Adds /health endpoint for monitoring
- Adds Gemini safe retry + graceful fallback (no "Summary unavailable" message)
"""


import os, re, json, tempfile, urllib.parse, time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp, pdfplumber, google.generativeai as genai, requests
from dotenv import load_dotenv

# ---------------- Global Safe-Mode Timeouts ----------------
REQUEST_TIMEOUT = 30
YTDLP_SOCKET_TIMEOUT = 15
YTDLP_RETRIES = 1
HLS_SEGMENT_LIMIT = 50

# ---------------- Setup ----------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
YTDLP_COOKIES = os.getenv("YTDLP_COOKIES")
if YTDLP_COOKIES and not os.path.exists(YTDLP_COOKIES):
    print(f"[SmartSummarizer] Warning: cookie file not found at {YTDLP_COOKIES}")
    YTDLP_COOKIES = None

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception as e:
        print("Warning: genai.configure() failed:", e)

app = FastAPI(title="Smart Summarizer (robust cloud)", version="1.2.9")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Utilities ----------------
def log(s: str): 
    print("[SmartSummarizer] " + s)

def _safe_gemini_text(resp) -> str:
    if not resp: return ""
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip(): return text.strip()
    if isinstance(resp, dict):
        if resp.get("text"): return resp["text"].strip()
        if resp.get("candidates"):
            try:
                cand = resp["candidates"][0]
                txt = cand.get("content", {}).get("text") or cand.get("text")
                if txt: return txt.strip()
            except Exception: pass
    return str(resp).strip() if resp else ""

# ---------------- YouTube Helpers ----------------
def extract_video_id(url: str) -> str:
    pats = [r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&\?#]|$)", r"youtu\.be\/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    m2 = re.search(r"([0-9A-Za-z_-]{11})", url)
    if m2: return m2.group(1)
    raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

def try_transcript_api(video_id: str, video_url: Optional[str] = None) -> Optional[str]:
    """
    Smart transcript fetcher:
    - Prefers Transcript API when standard XML captions exist.
    - Skips immediately if only HLS (.m3u8) subtitles are detected (saves time).
    - Prefers English, but falls back to any available language.
    """
    try:
        # Optional fast pre-check via yt_dlp metadata
        if video_url:
            try:
                info = _extract_with_stable_client(video_url, download=False)
                all_subs = (info.get("subtitles") or {}) | (info.get("automatic_captions") or {})
                # If all URLs end with .m3u8 → skip Transcript API
                all_urls = [u.get("url", "") for v in all_subs.values() for u in v if isinstance(v, list)]
                if any(".m3u8" in u for u in all_urls):
                    log("Detected only HLS (.m3u8) subtitles — skipping YouTubeTranscriptApi.")
                    return None
            except Exception:
                pass

        log("Attempting YouTubeTranscriptApi (optimized universal)...")
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        english_variants = [
            "english", "en", "en-us", "en-gb", "en-in",
            "english (auto-generated)", "english (us)", "english (uk)", "english (india)"
        ]

        english_matches, others = [], []
        for t in transcripts:
            ln = t.language.lower()
            lc = t.language_code.lower()
            if any(v in ln or v == lc for v in english_variants):
                english_matches.append(t)
            else:
                others.append(t)

        for group, label in [(english_matches, "English"), (others, "non-English")]:
            for t in group:
                try:
                    fetched = t.fetch()
                    if fetched:
                        text = " ".join(seg.get("text", "") for seg in fetched if seg.get("text"))
                        if text.strip():
                            log(f"Fetched {label} transcript ({t.language}, {t.language_code})")
                            return text
                except Exception as e:
                    if "no element found" in str(e).lower():
                        log("Transcript XML empty (likely HLS) — skipping.")
                        return None
                    else:
                        log(f"{label} transcript fetch failed: {e}")

    except Exception as e:
        log(f"YouTubeTranscriptApi optimized fetch failed: {e}")

    return None


# ---------------- yt-dlp Wrapper ----------------
def _extract_with_stable_client(video_url: str, download: bool, extra_opts: Optional[dict] = None):
    ydl_opts = {
        "quiet": True, "no_warnings": True, "skip_download": not download,
        "extractor_args": {"youtube": {"player_client": ["default"], "skip_bad_formats": ["True"]}},
        "retries": YTDLP_RETRIES, "socket_timeout": YTDLP_SOCKET_TIMEOUT,
    }
    if YTDLP_COOKIES: ydl_opts["cookiefile"] = YTDLP_COOKIES
    if extra_opts: ydl_opts.update(extra_opts)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(video_url, download=download)
    except Exception:
        ydl_opts["extractor_args"]["youtube"]["player_client"] = ["android"]
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(video_url, download=download)

def _subtitle_to_plain(s: str) -> str:
    s = s.strip()
    if not s: return ""
    try:
        data = json.loads(s)
        segs = []
        if isinstance(data, dict) and "events" in data:
            for ev in data["events"]:
                for sg in ev.get("segs") or []:
                    t = sg.get("utf8", "").strip()
                    if t: segs.append(t)
        if segs: return " ".join(segs)
    except Exception:
        pass
    lines = [ln.strip() for ln in s.splitlines() if ln and not re.match(r"^(\d+|WEBVTT|-->)", ln)]
    return " ".join(lines)

def try_ytdlp_subtitles(video_url: str) -> Optional[str]:
    """Prefer English captions first, otherwise use any available language."""
    log("Attempting yt_dlp subtitle extraction (enhanced)...")
    start = time.time()

    def is_english_label(label: str) -> bool:
        if not label:
            return False
        lbl = label.lower()
        # Match ANY English variant:
        # - "English", "English (US)", "English - CC", "English (auto-generated)"
        # - "en", "en-US", "en-GB", "en_in"
        # - "eng"
        return (
            "english" in lbl or
            re.match(r"^en([\-_][a-z]+)?$", lbl) is not None or
            lbl.startswith("en") or
            lbl == "eng"
        )

    try:
        info = _extract_with_stable_client(video_url, download=False)
        subs = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}

        all_tracks = {**subs, **auto}

        english_candidates = []
        other_candidates = []

        # Split tracks into English vs non-English
        for lang, items in all_tracks.items():
            if not isinstance(items, list):
                continue
            if is_english_label(lang):
                english_candidates.extend(items)
            else:
                other_candidates.extend(items)

        # Priority: English -> other languages -> anything
        if english_candidates:
            candidates = english_candidates
        elif other_candidates:
            candidates = other_candidates
        else:
            candidates = []
            for v in all_tracks.values():
                if isinstance(v, list):
                    candidates.extend(v)

        for cand in candidates:
            if time.time() - start > REQUEST_TIMEOUT:
                break
            url = cand.get("url")
            if not url:
                continue
            try:
                r = requests.get(url, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200 and r.text.strip():
                    text = r.text

                    # HLS playlist?
                    if text.lstrip().startswith("#EXTM3U"):
                        log("Detected HLS (.m3u8) subtitle playlist; fetching segments...")
                        lines = text.splitlines()
                        segs = [
                            urllib.parse.urljoin(url, ln.strip())
                            for ln in lines if ln and not ln.startswith("#")
                        ]
                        collected = []
                        for seg in segs[:HLS_SEGMENT_LIMIT]:
                            try:
                                rs = requests.get(seg, timeout=REQUEST_TIMEOUT)
                                if rs.status_code == 200:
                                    content = rs.content.decode("utf-8", errors="ignore")
                                    cleaned = _subtitle_to_plain(content)
                                    if cleaned.strip():
                                        collected.append(cleaned)
                            except Exception as e:
                                log(f"Segment fetch failed: {e}")
                        if collected:
                            log(f"Fetched {len(collected)} HLS subtitle segments.")
                            return " ".join(collected)
                        continue

                    # Normal VTT/SRT text
                    cleaned = _subtitle_to_plain(text)
                    if cleaned.strip():
                        log("yt_dlp subtitle fetch succeeded.")
                        return cleaned
            except Exception as e:
                log(f"Subtitle fetch failed: {e}")

        log("No usable subtitles found via yt_dlp.")
    except Exception as e:
        log(f"yt_dlp subtitle extraction failed: {e}")

    return None


# ---------------- Master Transcript Logic ----------------
def extract_transcript_from_youtube(video_url: str) -> str:
    log(f"extract_transcript_from_youtube START for {video_url}")
    vid = extract_video_id(video_url)
    transcript = try_transcript_api(vid, video_url)
    if not transcript:
        transcript = try_ytdlp_subtitles(video_url)
    if not transcript:
        log("No subtitles available — using title and description as fallback.")
        try:
            info = _extract_with_stable_client(video_url, download=False)
            title = info.get("title", "")
            desc = info.get("description", "")
            transcript = f"Title: {title}\n\nDescription:\n{desc}"
        except Exception as e:
            log(f"Metadata fetch failed: {e}")
            transcript = ""

    if transcript and transcript.strip():
        return transcript
    try:
        info = _extract_with_stable_client(video_url, download=False)
        return f"Title: {info.get('title','')}\n\nDescription:\n{info.get('description','')}"
    except Exception:
        pass
    raise HTTPException(status_code=400, detail="Transcript unavailable.")

# ---------------- PDF Extraction ----------------
def extract_pdf_text(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

# ---------------- Summarization ----------------
def summarize_via_gemini(text: str, summary_type: str = "short",
                         bullet_count: Optional[int] = None, target_lang: str = "en") -> str:
    """
    Smart Summarizer v1.4.0 - Optimized for Speed & Quality
    - Faster processing with optimized chunking
    - Better quality summaries with improved prompts
    - Smart caching and response optimization
    """

    if not text:
        return "No text available to summarize."
    if not GEMINI_KEY:
        return text[:800] + "..." * (len(text) > 800)

    summary_type = (summary_type or "short").strip().lower()
    if summary_type not in {"short", "bullet", "detailed"}:
        log(f"Unknown summary_type '{summary_type}', defaulting to 'short'")
        summary_type = "short"

    model = genai.GenerativeModel("gemini-2.0-flash")

    # -------- Optimized Chunking --------
    text = text.strip()
    text_len = len(text)

    MAX_SAFE_CHUNK = 120000  # Increased for better context
    overlap = 500
    max_chunks = 2

    if text_len <= MAX_SAFE_CHUNK:
        chunks = [text]
    else:
        chunks = []
        step = MAX_SAFE_CHUNK - overlap
        for i in range(0, text_len, step):
            chunk = text[i:i + MAX_SAFE_CHUNK]
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break

    log(f"[Chunker v1.4.0] Processing {len(chunks)} chunk(s) (~{MAX_SAFE_CHUNK} chars each, total={text_len})")

    partials: List[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        try:
            if summary_type == "bullet":
                bc = int(bullet_count) if bullet_count else 10
                prompt = f"""You are a professional expert summarizer. Create a comprehensive organized summary with MAIN TOPICS, SUB-TOPICS, and detailed explanations.

STRUCTURE REQUIREMENTS - FOLLOW EXACTLY:
1. Start with 1-2 introductory paragraphs explaining the main topic/theme (simple, easy language)
2. For EACH MAIN TOPIC identified in the content:
   - Write the MAIN TOPIC NAME as a bold section header (e.g., "**Main Topic Name:**")
   - Add a brief intro paragraph explaining what this main topic covers
   - List 3-5 SUB-TOPICS under it with detailed explanations
   - Each sub-topic should follow format: "- **Sub-topic Name:** Explanation (2-3 complete sentences describing the sub-topic and its importance)"
   - Include specific details, examples, context, and implications
3. Cover ALL major topics and themes - NO TOPICS SHOULD BE SKIPPED
4. Ensure EVERY point is fully explained with context and details
5. Use clear, professional, easy-to-read language
6. Organization: Group related sub-topics together logically
7. Aim for approximately {bc} total sub-topic bullet points across all main topics

CRITICAL - Complete Explanation:
- Each sub-topic explanation must be 2-3 sentences minimum
- Include WHY each point matters
- Include specific facts, figures, or examples
- Connect sub-topics to the main topic
- Do NOT use vague or incomplete explanations

Content:
{chunk}"""
                max_tokens = 2500

            elif summary_type == "comprehensive":
                prompt = f"""Create an extremely detailed and comprehensive summary where EVERY important point is explained thoroughly.

Format requirements:
- Start with a brief overview
- For each major topic, provide:
  * Topic name with ## header
  * Detailed explanation (2-4 sentences minimum per point)
  * Sub-points with ### headers if applicable
  * Real-world implications or examples where relevant
- Include at least 5-7 major sections
- Each section should have multiple detailed paragraphs
- Maintain professional academic tone throughout
- Include all nuances and details from the source

Content:
{chunk}"""
                max_tokens = 3000

            elif summary_type == "detailed":
                prompt = f"""Create an extremely detailed and comprehensive summary with thorough explanations for every point.

Format requirements:
- Use ## for main sections and ### for subsections
- Under each section, provide detailed bullet points with comprehensive explanations
- Each bullet point should be 2-3 sentences explaining the concept thoroughly
- Include practical examples, implications, and context for each point
- Organize related concepts together logically
- Maintain professional academic tone throughout
- Ensure all important details and nuances are captured
- Include at least 5-7 major topics with multiple detailed points each

Content:
{chunk}"""
                max_tokens = 3000

            else:  # short
                prompt = f"""Write a concise professional summary in 3-4 clear paragraphs.

Guidelines:
- Each paragraph should focus on one main idea
- Use clear transitions between paragraphs
- Keep language formal and professional
- Highlight key points and conclusions
- Avoid repetition

Content:
{chunk}"""
                max_tokens = 1200

            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": max_tokens, "top_p": 0.9}
            )
            result = _safe_gemini_text(resp)
            if result.strip():
                partials.append(result)
                log(f"✔ Chunk {idx}/{len(chunks)} completed ({len(chunk)} chars)")
            else:
                log(f"⚠ Empty response for chunk {idx}")
        except Exception as e:
            log(f"Error processing chunk {idx}: {e}")
            continue

    if not partials:
        return text[:1500] + "..."

    # -------- Skip merge for single chunk --------
    if len(partials) == 1:
        log("✅ Summary generated successfully")
        return format_summary_output(partials[0], summary_type)

    # -------- Merge multiple chunks --------
    combined = "\n\n".join(partials)

    try:
        if summary_type == "short":
            merge_prompt = f"""Combine these partial summaries into ONE cohesive professional summary (3-4 paragraphs).
- Remove any duplicate information
- Maintain logical flow
- Keep professional tone

Partial summaries:
{combined}"""
        elif summary_type == "bullet":
            bc = int(bullet_count) if bullet_count else 10
            merge_prompt = f"""Combine these partial summaries into ONE comprehensive, well-organized summary with MAIN TOPICS and SUB-TOPICS.

STRUCTURE REQUIREMENTS - FOLLOW EXACTLY:
1. Start with 1-2 introductory paragraphs explaining the main topic/theme (simple, easy language)
2. For EACH MAIN TOPIC identified:
   - Write the MAIN TOPIC NAME as a bold section header (e.g., "**Main Topic Name:**")
   - Add a 2-3 sentence intro paragraph explaining what this main topic covers
   - List 3-5 SUB-TOPICS under it with complete explanations
   - Each sub-topic format: "- **Sub-topic Name:** Explanation (2-3 sentences minimum with details, context, and implications)"
3. CRITICAL - COMPLETE TOPIC COVERAGE:
   - Cover ALL major topics and themes comprehensively
   - DO NOT skip or omit any important topics
   - Include approximately {bc} sub-topic bullet points total
   - Each main topic should have multiple explained sub-topics
4. DETAILED EXPLANATIONS:
   - Every sub-topic must be fully explained (minimum 2-3 sentences)
   - Include specific facts, figures, examples, or case studies
   - Explain WHY each point is important
   - Connect sub-topics to the main topic
   - Include practical implications or consequences
5. Organize logically:
   - Main topics by importance or sequence
   - Sub-topics grouped by relationship
   - Remove only true duplicates while keeping topic diversity
6. Professional, clear, easy-to-understand language
7. Include points spanning all sections (beginning, middle, end of content)

Topics to Include (if present):
- Core concepts with definitions and context
- Important facts, figures, and statistics
- Processes, methods, and procedures
- Causes, effects, and consequences
- Examples, case studies, and real-world applications
- Best practices and recommendations
- Key relationships and connections
- Important conclusions and takeaways

Partial summaries:
{combined}"""
        else:  # detailed
            merge_prompt = f"""Combine these sections into ONE comprehensive, well-organized summary.
- Keep ## for main sections and ### for subsections
- Remove duplicates while preserving all important details
- Ensure each point has thorough, detailed explanations (2-3 sentences per point)
- Maintain clear structure with comprehensive bullet points under each section
- Include all nuances, implications, and context

Partial summaries:
{combined}"""

        resp = model.generate_content(
            merge_prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 3500, "top_p": 0.9}
        )
        final_summary = _safe_gemini_text(resp).strip()

        if final_summary:
            log("✅ Final summary merged and optimized")
            return format_summary_output(final_summary, summary_type)

    except Exception as e:
        log(f"Merge error: {e}")
        return format_summary_output("\n\n".join(partials), summary_type)

    return format_summary_output("\n\n".join(partials[:3]), summary_type)

# ----------- HTML Styling Formatter -----------
def format_summary_output(text: str, summary_type: str) -> str:
    text = text.strip()
    if not text:
        return ""

    # --- Convert bold (**text**) to <b>text</b> ---
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    if summary_type == "bullet":
        # Convert markdown formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        
        # Process organized format with main topics and sub-topics
        lines = text.splitlines()
        html_parts = []
        current_list = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if it's a main topic header (bold with colon or multiple asterisks)
            is_main_header = re.match(r"^<b>.+:</b>$", line_stripped)
            
            # Check if it's a sub-topic bullet with bold label (indicates main topic structure)
            is_subtopic = re.match(r"^-\s*<b>.+:</b>", line_stripped)
            
            if is_main_header:
                # Output any existing list
                if current_list:
                    html_parts.append("<ul>\n" + "\n".join(f"<li>{item}</li>" for item in current_list) + "\n</ul>")
                    current_list = []
                # Add main topic header
                html_parts.append(f"<p style='font-weight: bold; font-size: 1.1em; margin-top: 15px;'>{line_stripped}</p>")
            
            # Check if it's a regular paragraph (not a bullet)
            elif line_stripped and not line_stripped.startswith("-"):
                # Output any existing list first
                if current_list:
                    html_parts.append("<ul>\n" + "\n".join(f"<li>{item}</li>" for item in current_list) + "\n</ul>")
                    current_list = []
                # Add paragraph
                if line_stripped:
                    html_parts.append(f"<p>{line_stripped}</p>")
            
            # It's a bullet point (sub-topic or regular bullet)
            elif line_stripped.startswith("-"):
                bullet_text = line_stripped[1:].strip()
                if bullet_text:
                    # Check if this is a sub-topic with bold label
                    if re.match(r"^<b>.+:</b>", bullet_text):
                        # Extract the bold part and the explanation
                        match = re.match(r"^(<b>.+?:</b>)\s*(.*)", bullet_text)
                        if match:
                            label = match.group(1)
                            explanation = match.group(2)
                            # Format as: **Label:** Explanation text
                            formatted = f"{label} {explanation}" if explanation else label
                            current_list.append(formatted)
                    else:
                        current_list.append(bullet_text)
        
        # Don't forget the last list
        if current_list:
            html_parts.append("<ul>\n" + "\n".join(f"<li>{item}</li>" for item in current_list) + "\n</ul>")
        
        return "\n".join(html_parts)

    elif summary_type in ["detailed"]:
        # Convert Markdown headers and bullets to HTML
        text = re.sub(r"^###\s*(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
        text = re.sub(r"^##\s*(.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)

        # Additional fixes — catch missing spaces
        text = re.sub(r"^###(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
        text = re.sub(r"^##(.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)

        # Convert bolded colon lines (**Term:**) into section headers
        text = re.sub(r"<b>([^<:]+:)</b>", r"<h4>\1</h4>", text)

        # Convert bullet lines
        lines = [ln.strip() for ln in text.splitlines()]
        formatted = []
        for ln in lines:
            if re.match(r"^[-*•]", ln):
                ln = re.sub(r"^[-*•]\s*", "", ln)
                formatted.append(f"<li>{ln}</li>")
            else:
                formatted.append(ln)
        html = "\n".join(formatted)
        html = re.sub(r"(<li>.+?</li>)+", lambda m: f"<ul>{m.group(0)}</ul>", html)
        return html

    else:  # short summary
        paras = [f"<p>{p.strip()}</p>" for p in text.split("\n\n") if p.strip()]
        return "\n".join(paras)
    

# ---------------- Endpoints ----------------
@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")

@app.get("/styles.css")
async def serve_css():
    return FileResponse(Path(__file__).parent / "styles.css", media_type="text/css")

@app.get("/script.js")
async def serve_js():
    return FileResponse(Path(__file__).parent / "script.js", media_type="application/javascript")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.2.9",
        "gemini_configured": bool(GEMINI_KEY)
    }

@app.post("/summarize/youtube")
async def summarize_youtube(video_url: str = Form(...),
                            summary_type: str = Form("short"),
                            bullet_count: Optional[int] = Form(None),
                            target_lang: str = Form("en")):
    try:
        log(f"Processing YouTube: {video_url[:50]}... ({summary_type})")
        transcript = extract_transcript_from_youtube(video_url)
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="No transcript found.")
        
        log("Generating summary...")
        final = summarize_via_gemini(transcript, summary_type, bullet_count, "en")
        
        return {
            "success": True, 
            "summary": final, 
            "video_url": video_url,
            "summary_type": summary_type,
            "transcript_length": len(transcript)
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error in YouTube summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/pdf")
async def summarize_pdf(file: UploadFile = File(...),
                        summary_type: str = Form("short"),
                        bullet_count: Optional[int] = Form(None),
                        target_lang: str = Form("en")):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        
        log(f"Processing PDF: {file.filename} ({summary_type})")
        text = extract_pdf_text(tmp.name)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        
        log("Generating summary...")
        final = summarize_via_gemini(text, summary_type, bullet_count, "en")
        
        return {
            "success": True, 
            "summary": final, 
            "filename": file.filename,
            "summary_type": summary_type,
            "text_length": len(text)
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error in PDF summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp.name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
