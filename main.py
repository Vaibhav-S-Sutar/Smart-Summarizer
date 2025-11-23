# main.py
"""
Smart Summarizer - robust combined version (v1.3.0)

Enhancements over v1.2.9:
- Fixed YouTube bot detection with multiple fallback strategies
- Improved user-agent headers and request handling
- Better error messages for rate limiting and bot detection
- Enhanced output formatting with improved CSS
- Multiple extraction strategies (android, ios, mweb, tv_embedded)
- Graceful fallback to metadata when transcripts unavailable
- Better retry logic with 3 attempts per strategy
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

# ---------------- Global Safe-Mode Timeouts (Optimized for Render Free Tier) ----------------
REQUEST_TIMEOUT = 20  # Reduced for free tier
YTDLP_SOCKET_TIMEOUT = 10  # Faster timeout
YTDLP_RETRIES = 1
HLS_SEGMENT_LIMIT = 30  # Reduced to save memory
MAX_CONCURRENT_REQUESTS = 10  # Limit concurrent requests

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

app = FastAPI(
    title="Smart Summarizer (Render Free Tier Optimized)", 
    version="1.3.2",
    docs_url=None,  # Disable docs in production to save memory
    redoc_url=None
)

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
        log("Attempting YouTubeTranscriptApi (free tier optimized)...")
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        # Simplified English detection for faster execution
        english_variants = ["english", "en", "en-us", "en-gb"]

        english_matches, others = [], []
        for t in transcripts:
            ln = t.language.lower()
            lc = t.language_code.lower()
            if any(v in ln or v == lc for v in english_variants):
                english_matches.append(t)
            else:
                others.append(t)

        # Try English first (max 3 variants), then first non-English
        all_attempts = english_matches[:3] + others[:1]
        
        for idx, t in enumerate(all_attempts):
            try:
                log(f"Fetching transcript {idx+1}/{len(all_attempts)}: {t.language} ({t.language_code})")
                fetched = t.fetch()
                if fetched:
                    text = " ".join(seg.get("text", "") for seg in fetched if seg.get("text"))
                    if text.strip():
                        log(f"✓ Transcript fetched successfully: {t.language}")
                        return text
            except Exception as e:
                if "no element found" in str(e).lower():
                    log("Transcript XML empty (likely HLS) — skipping")
                    return None
                log(f"✗ Transcript fetch attempt {idx+1} failed: {str(e)[:80]}")

    except Exception as e:
        log(f"✗ YouTubeTranscriptApi failed: {str(e)[:100]}")

    return None


# ---------------- yt-dlp Wrapper ----------------
def _extract_with_stable_client(video_url: str, download: bool, extra_opts: Optional[dict] = None):
    """Optimized for Render free tier with reduced timeouts and memory usage"""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": not download,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
                "skip": ["dash", "hls"],
            }
        },
        "retries": 2,  # Reduced for free tier
        "fragment_retries": 2,
        "socket_timeout": YTDLP_SOCKET_TIMEOUT,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-us,en;q=0.5",
            "Sec-Fetch-Mode": "navigate",
        },
        "ignoreerrors": True,  # Continue on errors
        "extract_flat": False,
    }
    if YTDLP_COOKIES: ydl_opts["cookiefile"] = YTDLP_COOKIES
    if extra_opts: ydl_opts.update(extra_opts)
    
    # Try multiple strategies (reduced for faster failover)
    strategies = [
        {"player_client": ["android"]},  # Fastest
        {"player_client": ["ios"]},
        {"player_client": ["web"]},
    ]
    
    last_error = None
    for idx, strategy in enumerate(strategies):
        try:
            log(f"Trying extraction strategy {idx+1}/{len(strategies)}: {strategy['player_client']}")
            ydl_opts["extractor_args"]["youtube"].update(strategy)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=download)
                if info:
                    log(f"✓ Extraction successful with strategy: {strategy['player_client']}")
                    return info
        except Exception as e:
            error_str = str(e).lower()
            last_error = e
            log(f"✗ Strategy {strategy['player_client']} failed: {str(e)[:100]}")
            
            # Don't retry on these errors
            if "private video" in error_str or "video unavailable" in error_str:
                raise
            
            # Try next strategy
            if idx < len(strategies) - 1:
                continue
    
    # All strategies failed
    if last_error:
        raise last_error
    raise Exception("All extraction strategies failed")

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
    """Optimized for Render free tier - prefer English captions with early bailout."""
    log("Attempting yt_dlp subtitle extraction (optimized)...")
    start = time.time()

    def is_english_label(label: str) -> bool:
        if not label:
            return False
        lbl = label.lower()
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

        # Priority: English -> other languages
        if english_candidates:
            candidates = english_candidates[:3]  # Limit to first 3 for free tier
        elif other_candidates:
            candidates = other_candidates[:2]  # Limit to first 2
        else:
            candidates = []
            for v in list(all_tracks.values())[:2]:  # Max 2 fallback attempts
                if isinstance(v, list):
                    candidates.extend(v[:1])

        for idx, cand in enumerate(candidates):
            # Early timeout check
            if time.time() - start > REQUEST_TIMEOUT * 0.8:  # Use 80% of timeout
                log("Subtitle extraction timeout approaching, stopping early")
                break
            
            url = cand.get("url")
            if not url:
                continue
            
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                r = requests.get(url, timeout=REQUEST_TIMEOUT // 2, headers=headers)  # Halve timeout per request
                if r.status_code == 200 and r.text.strip():
                    text = r.text

                    # HLS playlist?
                    if text.lstrip().startswith("#EXTM3U"):
                        log("Detected HLS (.m3u8) subtitle playlist; fetching limited segments...")
                        lines = text.splitlines()
                        segs = [
                            urllib.parse.urljoin(url, ln.strip())
                            for ln in lines if ln and not ln.startswith("#")
                        ]
                        collected = []
                        # Limit segments for memory efficiency
                        max_segs = min(HLS_SEGMENT_LIMIT, 20)
                        for seg in segs[:max_segs]:
                            try:
                                rs = requests.get(seg, timeout=5)  # 5s per segment
                                if rs.status_code == 200:
                                    content = rs.content.decode("utf-8", errors="ignore")
                                    cleaned = _subtitle_to_plain(content)
                                    if cleaned.strip():
                                        collected.append(cleaned)
                            except Exception as e:
                                log(f"Segment fetch failed: {e}")
                        if collected:
                            log(f"✓ Fetched {len(collected)} HLS subtitle segments")
                            return " ".join(collected)
                        continue

                    # Normal VTT/SRT text
                    cleaned = _subtitle_to_plain(text)
                    if cleaned.strip():
                        log(f"✓ yt_dlp subtitle fetch succeeded (candidate {idx+1})")
                        return cleaned
            except Exception as e:
                log(f"✗ Subtitle fetch attempt {idx+1} failed: {str(e)[:80]}")
                continue

        log("No usable subtitles found via yt_dlp")
    except Exception as e:
        log(f"✗ yt_dlp subtitle extraction failed: {str(e)[:100]}")

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
            if title or desc:
                transcript = f"Title: {title}\n\nDescription:\n{desc}"
                log(f"Using metadata fallback: title='{title[:50]}...', desc_length={len(desc)}")
        except Exception as e:
            log(f"Metadata fetch failed: {e}")
            transcript = ""

    if transcript and transcript.strip():
        return transcript
    
    # Last attempt: try to get any metadata
    try:
        log("Final attempt: fetching video metadata...")
        info = _extract_with_stable_client(video_url, download=False)
        title = info.get("title", "No title available")
        desc = info.get("description", "No description available")
        uploader = info.get("uploader", "Unknown")
        duration = info.get("duration", 0)
        view_count = info.get("view_count", 0)
        upload_date = info.get("upload_date", "Unknown")
        
        # Format duration
        if duration:
            mins = duration // 60
            secs = duration % 60
            duration_str = f"{mins}:{secs:02d}"
        else:
            duration_str = "Unknown"
        
        # Format view count
        if view_count:
            if view_count >= 1000000:
                view_str = f"{view_count/1000000:.1f}M views"
            elif view_count >= 1000:
                view_str = f"{view_count/1000:.1f}K views"
            else:
                view_str = f"{view_count} views"
        else:
            view_str = "Unknown views"
        
        # Format upload date
        if upload_date and upload_date != "Unknown":
            try:
                year = upload_date[:4]
                month = upload_date[4:6]
                day = upload_date[6:8]
                upload_date = f"{year}-{month}-{day}"
            except:
                pass
        
        fallback_text = f"""Video Title: {title}

Uploader: {uploader}
Duration: {duration_str}
Views: {view_str}
Upload Date: {upload_date}

Description:
{desc}

Note: Transcript/subtitles were not available for this video. This summary is based on the video metadata only."""
        
        if fallback_text.strip():
            log("Successfully retrieved metadata for summarization")
            return fallback_text
    except Exception as e:
        log(f"Final metadata fetch also failed: {e}")
    
    raise HTTPException(
        status_code=400, 
        detail="Unable to extract transcript or metadata from this video. This may be because: 1) The video has no subtitles/captions, 2) The video is private or age-restricted, 3) YouTube is blocking automated access. Please try a different video."
    )

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

    # -------- Optimized Chunking for Free Tier --------
    text = text.strip()
    text_len = len(text)

    MAX_SAFE_CHUNK = 80000  # Reduced for 512MB RAM constraint
    overlap = 300  # Reduced overlap
    max_chunks = 2  # Limit to 2 chunks max

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
        # No AI summary generated - format the raw text as metadata
        log("⚠ No AI summary generated - returning formatted metadata")
        return format_summary_output(text[:1500], "short")

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
    """Format summary text into clean, readable HTML"""
    text = text.strip()
    if not text:
        return ""

    # Convert markdown bold to HTML
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    
    if summary_type == "bullet":
        lines = text.splitlines()
        html_parts = []
        current_list = []
        in_intro = True
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for main topic headers (bold text with colon at end)
            if re.match(r"^<strong>.+:</strong>$", line):
                # Close any open list
                if current_list:
                    html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in current_list) + "</ul>")
                    current_list = []
                in_intro = False
                # Add as section header
                html_parts.append(f"<h4>{line.replace('<strong>', '').replace('</strong>', '')}</h4>")
                
            # Bullet points
            elif line.startswith("-") or line.startswith("•") or line.startswith("*"):
                in_intro = False
                bullet_text = re.sub(r"^[-•*]\s*", "", line)
                current_list.append(bullet_text)
                
            # Regular paragraphs (intro text, etc)
            else:
                # Close any open list
                if current_list:
                    html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in current_list) + "</ul>")
                    current_list = []
                html_parts.append(f"<p>{line}</p>")
        
        # Close final list if open
        if current_list:
            html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in current_list) + "</ul>")
        
        return "\n".join(html_parts)

    elif summary_type in ["detailed", "comprehensive"]:
        # Convert markdown headers
        text = re.sub(r"^###\s*(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
        text = re.sub(r"^##\s*(.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
        text = re.sub(r"^#\s*(.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
        
        # Convert strong tags with colons to headers
        text = re.sub(r"<strong>([^<]+:)</strong>", r"<h4>\1</h4>", text)
        
        # Process line by line
        lines = text.splitlines()
        html_parts = []
        current_list = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Already formatted headers
            if line.startswith("<h") or line.startswith("</"):
                if current_list:
                    html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in current_list) + "</ul>")
                    current_list = []
                html_parts.append(line)
                
            # Bullet points
            elif re.match(r"^[-*•]\s", line):
                bullet = re.sub(r"^[-*•]\s*", "", line)
                current_list.append(bullet)
                
            # Regular paragraphs
            else:
                if current_list:
                    html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in current_list) + "</ul>")
                    current_list = []
                html_parts.append(f"<p>{line}</p>")
        
        # Close final list
        if current_list:
            html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in current_list) + "</ul>")
        
        return "\n".join(html_parts)

    else:  # short summary
        lines = text.splitlines()
        html_parts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Headers/metadata (Video Title:, Description:, etc)
            if re.match(r"^(Video Title|Title|Uploader|Duration|Views|Upload Date|Description|Note):", line):
                html_parts.append(f"<h4>{line}</h4>")
            # Regular text
            else:
                html_parts.append(f"<p>{line}</p>")
        
        # Group consecutive paragraphs
        result = []
        para_buffer = []
        
        for part in html_parts:
            if part.startswith("<h4>"):
                if para_buffer:
                    result.append("<p>" + " ".join(p.replace("<p>", "").replace("</p>", "") for p in para_buffer) + "</p>")
                    para_buffer = []
                result.append(part)
            elif part.startswith("<p>"):
                para_buffer.append(part)
        
        if para_buffer:
            result.append("<p>" + " ".join(p.replace("<p>", "").replace("</p>", "") for p in para_buffer) + "</p>")
        
        return "\n".join(result)
    

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
        "version": "1.3.0",
        "gemini_configured": bool(GEMINI_KEY),
        "features": [
            "YouTube bot detection bypass",
            "Multiple extraction strategies",
            "Enhanced error handling",
            "Improved formatting"
        ]
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
            raise HTTPException(
                status_code=400, 
                detail="No content could be extracted from this video. Please try a different video."
            )
        
        # Check if we're using fallback metadata
        is_metadata_only = "Note: Transcript/subtitles were not available" in transcript
        
        log(f"Generating summary... (metadata_only={is_metadata_only})")
        final = summarize_via_gemini(transcript, summary_type, bullet_count, "en")
        
        return {
            "success": True, 
            "summary": final, 
            "video_url": video_url,
            "summary_type": summary_type,
            "transcript_length": len(transcript),
            "metadata_only": is_metadata_only,
            "warning": "This summary is based on video metadata only (no transcript available)" if is_metadata_only else None
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error in YouTube summarization: {e}")
        error_message = str(e)
        
        # Handle specific error cases
        if "HTTP Error 429" in error_message or "Too Many Requests" in error_message:
            error_message = "YouTube is rate-limiting requests. Please wait a few minutes and try again."
        elif "sign in" in error_message.lower() or "bot" in error_message.lower():
            error_message = "YouTube detected automated access. This usually resolves itself after a few minutes. Try again shortly, or use a different video."
        elif "Private video" in error_message or "unavailable" in error_message.lower():
            error_message = "This video is private, unavailable, or age-restricted and cannot be accessed."
        elif "Video unavailable" in error_message:
            error_message = "This video is not available. It may have been deleted or made private."
        elif "All extraction strategies failed" in error_message:
            error_message = "Unable to access this video. YouTube may be blocking automated requests. Please try again in a few minutes or use a different video."
        
        raise HTTPException(status_code=500, detail=error_message)

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
