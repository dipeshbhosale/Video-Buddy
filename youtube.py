import streamlit as st
import subprocess
import os
import tempfile
import requests
import logging # Added for robust logging
import time # Added for implementing delays
from urllib.parse import urlparse
#from faster_whisper import WhisperModel # Import faster-whisper
#from streamlit_extras.add_vertical_space import add_vertical_space
from sentence_transformers import SentenceTransformer # For RAG-style embedding
from sklearn.metrics.pairwise import cosine_similarity # For RAG retrieval

# --- Configuration ---

# --- Logging Configuration ---
# Configure logging to output informational messages and above.
# For debugging, you can change level to logging.DEBUG
# Consider adding a file handler for persistent logs if running in a non-interactive environment.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Variable Checks ---
GROQ_API_KEYS_FROM_ENV_STR = os.environ.get("GROQ_API_KEYS") # Comma-separated keys

# OPENAI_API_KEY_CONFIG = "YOUR_OPENAI_KEY" # No longer needed for transcription
# Fallback: A list of hardcoded keys. Using environment variables is strongly recommended.
GROQ_API_KEYS_HARDCODED = [
    "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", # Replace with your actual keys
    "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",
    "Ygsk_gkqXGnfyMNH36QYv3NxmWGdyb3FYjbqxg8gaT1CbezFybwAQXTI5",
    "Ygsk_gkqXGnfyMNH36QYv3NxmWGdyb3FYjbqxg8gaT1CbezFybwAQXTI5"
]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"
# Whisper Model Configuration
DEFAULT_WHISPER_MODEL_SIZE = "base.en" # Options: "tiny.en", "small.en", "medium.en", "large-v3"

# --- Whisper Configuration (Set based on your hardware) ---
# For CPU-only:
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8" # Good for CPU; other options: "int16", "float32"
# For GPU (NVIDIA with CUDA):
# WHISPER_DEVICE = "cuda"
# WHISPER_COMPUTE_TYPE = "float16" # Good for GPU; other options: "int8_float16", "float32"

# --- API Key Management ---
AVAILABLE_GROQ_API_KEYS = []
if GROQ_API_KEYS_FROM_ENV_STR:
    AVAILABLE_GROQ_API_KEYS = [key.strip() for key in GROQ_API_KEYS_FROM_ENV_STR.split(',') if key.strip()]
    logging.info(f"Loaded {len(AVAILABLE_GROQ_API_KEYS)} Groq API keys from environment variable GROQ_API_KEYS.")
else:
    AVAILABLE_GROQ_API_KEYS = [key for key in GROQ_API_KEYS_HARDCODED if key and not key.startswith("YOUR_")] # Filter out placeholders
    if AVAILABLE_GROQ_API_KEYS:
        logging.warning(
            f"GROQ_API_KEYS environment variable not set or empty. Using {len(AVAILABLE_GROQ_API_KEYS)} hardcoded API key(s). "
            "This is NOT recommended for production. Please set the GROQ_API_KEYS environment variable."
        )

CURRENT_API_KEY_INDEX = 0

# Whisper model display names and their actual values
WHISPER_MODEL_OPTIONS_DISPLAY = {
    "tiny.en (Fastest)": "tiny.en",
    "small.en (Faster)": "small.en",
    "base.en (Default)": "base.en",
    "medium.en (Slower, Better Quality)": "medium.en"
}

# --- Embedding Model Configuration (for RAG-style processing) ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # A popular, efficient sentence transformer
embedding_model = None # Will be loaded on first use
# --- RAG Configuration ---
RAG_TOP_K_CHUNKS = 3 # Number of most relevant chunks to retrieve for Q&A

# --- Chunking Configuration for Summarization ---
MAX_CHARS_PER_SUMMARY_CHUNK = 18000  # Max characters per transcript chunk for individual summarization
SUMMARY_CHUNK_OVERLAP_CHARS = 200    # Overlap between summary chunks to maintain context

# --- API Call Delay Configuration ---
API_CALL_DELAY_SECONDS = 2  # Increased default delay between API calls
RATE_LIMIT_DELAY_SECONDS = 10 # Longer delay after a 429 error
MAX_RETRIES_ON_429 = 1 # Number of retries for a single API call when a 429 is encountered


# --- Helper for API Key Rotation ---
def get_initial_groq_api_key():
    """Gets an initial API key for a new processing session using round-robin."""
    global CURRENT_API_KEY_INDEX
    if not AVAILABLE_GROQ_API_KEYS:
        raise ValueError("No Groq API keys are configured. Application cannot proceed.")
    key = AVAILABLE_GROQ_API_KEYS[CURRENT_API_KEY_INDEX]
    return key

# --- Functions ---

# Cache for different model sizes
transcription_models = {}

# --- Helper to ensure audio_path is cleaned up ---
class TempAudioFile:
    def __init__(self, path):
        self.path = path
    def __enter__(self):
        return self.path
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except Exception as e:
                logging.warning(f"Could not clean up temporary audio file: {self.path}. Error: {e}")
                # Keep user-facing warning if appropriate, or handle silently if preferred
                # For now, let's keep it to inform the user of potential leftover files.
                st.warning(f"Note: Could not automatically clean up a temporary audio file: {os.path.basename(self.path)}. You may need to delete it manually if it persists. Error: {e}")


def download_youtube_audio(url):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        output_path = tmp_file.name
    cmd = [
        "yt-dlp",
        # "--ffmpeg-location", FFMPEG_PATH,  # Commented out: yt-dlp should find ffmpeg in PATH
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0", # Best audio quality for MP3 conversion
        "-o", output_path,
        "--no-continue",      # Do not resume partially downloaded files (forces re-check of post-processing)
        "--force-overwrites", # Overwrite existing files, including during post-processing
        "-vU",                # Verbose output for debugging yt-dlp
        url
    ]
    try:
        logging.info(f"Executing yt-dlp command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8') # Specify encoding
        
        if result.stdout:
            logging.debug(f"yt-dlp stdout:\n{result.stdout}")
        if result.stderr: # yt-dlp often uses stderr for progress/info too
            logging.debug(f"yt-dlp stderr:\n{result.stderr}")

        # After successful run, check if the file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            error_message = f"Audio file was not created or is empty: {output_path}\n"
            if result.stdout: error_message += f"yt-dlp stdout:\n{result.stdout}\n"
            if result.stderr: error_message += f"yt-dlp stderr:\n{result.stderr}"
            logging.error(error_message)
            raise RuntimeError(error_message)

    except FileNotFoundError:
        error_message = (
            "yt-dlp command not found. Please ensure yt-dlp is installed and in your system's PATH. "
            "Visit https://github.com/yt-dlp/yt-dlp for installation instructions."
        )
        logging.error(error_message)
        # Re-raise as RuntimeError to be caught by the main error handler with a user-friendly message
        raise RuntimeError(error_message) from None
    except subprocess.CalledProcessError as e:
        error_message = f"yt-dlp failed with exit code {e.returncode}.\n"
        error_message += f"Command: {' '.join(e.cmd)}\n"
        error_message += f"Stdout:\n{e.stdout}\n"
        error_message += f"Stderr:\n{e.stderr}"
        logging.error(f"yt-dlp execution failed. Exit code: {e.returncode}")
        logging.error(f"Command: {' '.join(e.cmd)}")
        if e.stdout: logging.error(f"yt-dlp stdout:\n{e.stdout}")
        if e.stderr: logging.error(f"yt-dlp stderr:\n{e.stderr}")
        # Re-raise as RuntimeError to be caught by the main error handler
        raise RuntimeError(error_message) from e

    return output_path

def transcribe_with_local_whisper(audio_path, model_size=DEFAULT_WHISPER_MODEL_SIZE):
    global transcription_models
    if model_size not in transcription_models:
        # Model will be downloaded on first use if not already cached by faster-whisper
        st.info(f"Initializing transcription model '{model_size}' (this might take a moment on first run)...")
        try:
            loaded_model = WhisperModel(
                model_size,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            transcription_models[model_size] = loaded_model
            st.info(f"Transcription model '{model_size}' initialized.")
        except Exception as e:
            logging.error(f"Failed to load transcription model '{model_size}': {e}", exc_info=True)
            st.error(f"Failed to load transcription model '{model_size}': {e}")
            raise
    
    model_to_use = transcription_models[model_size]
    # You can adjust beam_size for a speed/accuracy tradeoff. Lower is faster.
    # VAD (Voice Activity Detection) can also speed up processing by skipping silence.
    # Example with VAD options (requires testing for optimal parameters):
    # segments, info = model_to_use.transcribe(audio_path, beam_size=5, language="en",
    #                                           vad_filter=True,
    #                                           vad_parameters=dict(min_silence_duration_ms=500))
    segments, info = model_to_use.transcribe(audio_path, beam_size=5, language="en") 

    return "".join(segment.text for segment in segments).strip()

# def summarize_with_groq(transcript): # Original function, will be replaced by chunked approach
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json",
#         "User-Agent": "VideoBuddyStreamlitApp/1.0" # Good practice to set a User-Agent
#     }
#     body = {
#         "model": GROQ_MODEL,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are an intelligent assistant that reads raw transcripts from YouTube videos and generates clear, concise, study-style bullet-point notes. "
#                     "Your primary goal is to help someone understand the video's content for review and learning without watching it. "
#                     "Focus on summarizing the main ideas, important facts, and actionable insights. Remove any filler, stutters, or irrelevant content from the transcript before summarizing. "
#                     "Always maintain the speaker's original intent. Avoid adding any information not present in the transcript and do not skip important context. "
#                     "Ensure each bullet point accurately reflects what was actually said. If something is uncertain or unclear in the transcript, explicitly mention this uncertainty rather than making assumptions or guessing. "
#                     "Keep the language simple, professional, and easy to understand.\n\n"
#                     "Use this format:\n"
#                     "- 🔹 Main topic or point\n"
#                     "- ➤ Sub-point or detail\n"
#                     "- ✅ Highlighted insight, key takeaway, or direct conclusion stated by the speaker\n"
#                     "- ⚠️ [Mention of uncertainty or unclear point from transcript, if any]"
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     "Please summarize the following YouTube video transcript into precise, factual bullet points. "
#                     "Do not generalize or invent details. Use the speaker's original ideas and phrasing as much as possible. "
#                     "If a list or step-by-step process is present, preserve the order. Avoid filler words, but retain all key points.\n\n"
#                     f"Transcript:\n\n{transcript}"
#                 )
#             }
#         ]
#     }

#     try:
#         response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=60) # Increased timeout
#         response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        
#         data = response.json()
#         # Validate response structure
#         if "choices" in data and data["choices"] and \
#            isinstance(data["choices"], list) and len(data["choices"]) > 0 and \
#            "message" in data["choices"][0] and \
#            "content" in data["choices"][0]["message"]:
#             return data["choices"][0]["message"]["content"]
#         else:
#             logging.error(f"Unexpected API response structure from Groq (summarize): {data}")
#             raise ValueError(f"Unexpected API response structure from Groq. Full response: {data}")
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Groq API request failed (summarize): {e}", exc_info=True)
#         raise RuntimeError(f"Failed to connect to Groq API for summarization: {e}") from e

# --- New Chunked Summarization Functions ---
def split_text_into_chunks(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start_index = 0
    text_length = len(text)

    while start_index < text_length:
        end_index = min(start_index + max_chars, text_length)
        chunks.append(text[start_index:end_index])
        
        if end_index == text_length:
            break 
        
        start_index = end_index - overlap_chars
        if start_index < 0: start_index = 0 # Should not happen if overlap < max_chars
        
        # Safety break for rare cases where start_index might not progress
        if start_index + max_chars <= end_index and chunks: # Check if new proposed chunk is fully within last added chunk
             if text[start_index:min(start_index + max_chars, text_length)] == chunks[-1][overlap_chars:] and overlap_chars > 0 :
                logging.warning(f"Chunk splitting might be stuck due to overlap. Advancing past current chunk. Start: {start_index}, End: {end_index}")
                start_index = end_index 

    return [c for c in chunks if c.strip()]

def summarize_transcript_chunk_with_groq(api_key_to_use, transcript_chunk, part_num, total_parts):
    time.sleep(API_CALL_DELAY_SECONDS) # Wait before making the call
    retries = 0
    # api_key_to_use = get_next_groq_api_key() # Key is now passed in
    headers = {
        "Authorization": f"Bearer {api_key_to_use}",
        "Content-Type": "application/json",
        "User-Agent": "VideoBuddyStreamlitApp/1.0" # Good practice to set a User-Agent
    }
    system_content = (
        f"You are an intelligent assistant. This is part {part_num} of {total_parts} of a YouTube video transcript. "
        "Generate clear, concise, study-style bullet-point notes for THIS PART of the video. "
        "Focus on the main ideas, important facts, and actionable insights within this specific segment. "
        "Remove filler or irrelevant content from this segment. Maintain the speaker's original intent. "
        "Do not add information not present in this segment. "
        "If something is uncertain or unclear in this segment, explicitly mention it. "
        "Keep the language simple and professional.\n\n"
        "Use this format for this part:\n"
        "- 🔹 Main topic or point in this segment\n"
        "- ➤ Sub-point or detail in this segment\n"
        "- ✅ Highlighted insight or key takeaway from this segment\n"
        "- ⚠️ [Mention of uncertainty or unclear point from this segment, if any]"
    )
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"Please summarize the following PARTIAL YouTube video transcript segment into precise, factual bullet points. "
                    f"This is part {part_num} of {total_parts}.\n\n"
                    f"Transcript Segment:\n\n{transcript_chunk}"
                )
            }
        ],
        "temperature": 0.3 
    }
    while retries <= MAX_RETRIES_ON_429:
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=90) 
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            
            data = response.json()
            # Validate response structure
            if "choices" in data and data["choices"] and \
               isinstance(data["choices"], list) and len(data["choices"]) > 0 and \
               "message" in data["choices"][0] and \
               "content" in data["choices"][0]["message"]:
                return data["choices"][0]["message"]["content"] # Success
            else:
                logging.error(f"Unexpected API response structure from Groq (chunk summarize part {part_num}): {data}")
                return f"[Error summarizing part {part_num}: Unexpected API response]" # Non-retryable error
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            key_identifier = f"...{api_key_to_use[-4:]}" if len(api_key_to_use) > 4 else api_key_to_use
            if status_code == 429 and retries < MAX_RETRIES_ON_429:
                logging.warning(f"Rate limit hit (429) for API key ending with '{key_identifier}' on chunk {part_num}. Retry {retries + 1}/{MAX_RETRIES_ON_429}. Waiting {RATE_LIMIT_DELAY_SECONDS}s.")
                time.sleep(RATE_LIMIT_DELAY_SECONDS)
                retries += 1
                # Optional: Could try switching to the next key here if retrying with the same key fails repeatedly.
                # For now, we retry with the same key.
            elif status_code == 401:
                logging.error(f"Unauthorized (401) for API key ending with '{key_identifier}' on chunk {part_num}. This key may be invalid or revoked.")
                return f"[Error summarizing part {part_num}: API request failed - {http_err}]" # Non-retryable auth error
            else: # Other HTTP errors
                logging.error(f"Groq API HTTP error (chunk summarize part {part_num}): {http_err}", exc_info=True)
                return f"[Error summarizing part {part_num}: API request failed - {http_err}]" # Non-retryable
        except requests.exceptions.RequestException as e:
            logging.error(f"Groq API request failed (chunk summarize part {part_num}): {e}", exc_info=True)
            return f"[Error summarizing part {part_num}: API request failed - {e}]" # Non-retryable
        except Exception as e:
            logging.error(f"An unexpected error occurred while summarizing chunk part {part_num}: {e}", exc_info=True)
            return f"[Error summarizing part {part_num}: Unexpected error - {e}]" # Non-retryable
    # If all retries fail for 429
    key_identifier = f"...{api_key_to_use[-4:]}" if len(api_key_to_use) > 4 else api_key_to_use
    logging.error(f"Max retries reached for 429 error on chunk {part_num} with key ending '{key_identifier}'.")
    return f"[Error summarizing part {part_num}: API request failed - Max retries for 429 error]"

def combine_summaries_with_groq(api_key_to_use, list_of_chunk_summaries, original_video_url="the video"):
    time.sleep(API_CALL_DELAY_SECONDS) # Wait before making the call
    # api_key_to_use = get_next_groq_api_key() # Key is now passed in
    if not list_of_chunk_summaries:
        return "No summaries were generated to combine."

    # Filter out error messages from chunk summaries before combining
    valid_chunk_summaries = [s for s in list_of_chunk_summaries if not s.startswith("[Error summarizing part")]
    if not valid_chunk_summaries:
        return "All parts failed to summarize. Cannot combine."
    if len(valid_chunk_summaries) == 1 and len(list_of_chunk_summaries) == 1: # Only one chunk, and it was successful
        logging.info("Only one successful chunk summary, returning it directly without combination call.")
        return valid_chunk_summaries[0]

    combined_text_for_prompt = "\n\n---\n\n".join(
        f"Summary of Part {i+1}:\n{summary_text}" # Use original index for user understanding
        for i, summary_text in enumerate(list_of_chunk_summaries) # Show errors too if any, for context
    )

    headers = {
        "Authorization": f"Bearer {api_key_to_use}",
        "Content-Type": "application/json",
        "User-Agent": "VideoBuddyStreamlitApp/1.0"
    }
    system_content = (
        "You are an expert summarizer. You have been provided with several partial summaries from different segments of a single YouTube video. "
        "Some parts might indicate an error during their summarization; if so, acknowledge that information for that part is missing or incomplete. "
        "Your task is to synthesize these partial summaries into ONE cohesive, comprehensive, and well-structured set of bullet-point notes that represents the ENTIRE video. "
        "Eliminate redundancy across the valid partial summaries, ensure a logical flow, and maintain the original study-note style. "
        "The final output should read as if it was a single summary of the whole video. "
        "Focus on the main ideas, important facts, and actionable insights from the entire video, based on the successfully summarized parts. "
        "Preserve the original bullet-point formatting style (🔹, ➤, ✅, ⚠️) where appropriate for the combined summary.\n\n"
        "Structure the final summary logically, perhaps chronologically or thematically based on the content of the partial summaries."
    )
    user_content = (
        f"Please combine the following partial summaries from the video ({original_video_url}) into a single, coherent set of notes. "
        f"If a part indicates an error, note that its content might be missing:\n\n"
        f"{combined_text_for_prompt}\n\n"
        "Final Combined Summary:"
    )

    if len(user_content) > 28000: # Heuristic to prevent payload too large for the combination call
        logging.warning(f"Combined chunk summaries are too long for a single combination call ({len(user_content)} chars). Returning concatenated summaries with error notes.")
        return f"The video was summarized in parts. Due to the combined length, a final synthesis was not performed. Individual part summaries (including any errors):\n\n{combined_text_for_prompt}"

    body = {"model": GROQ_MODEL, "messages": [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}], "temperature": 0.2}
    retries = 0
    while retries <= MAX_RETRIES_ON_429:
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=120)
            response.raise_for_status()
            data = response.json()
            if "choices" in data and data["choices"] and isinstance(data["choices"], list) and len(data["choices"]) > 0 and "message" in data["choices"][0] and "content" in data["choices"][0]["message"]:
                return data["choices"][0]["message"]["content"]
            else:
                logging.error(f"Unexpected API response structure from Groq (combine summaries): {data}")
                return f"[Error combining summaries: Unexpected API response]\n\n{combined_text_for_prompt}"
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            key_identifier = f"...{api_key_to_use[-4:]}" if len(api_key_to_use) > 4 else api_key_to_use
            if status_code == 429 and retries < MAX_RETRIES_ON_429:
                logging.warning(f"Rate limit hit (429) for API key ending with '{key_identifier}' on combining summaries. Retry {retries + 1}/{MAX_RETRIES_ON_429}. Waiting {RATE_LIMIT_DELAY_SECONDS}s.")
                time.sleep(RATE_LIMIT_DELAY_SECONDS)
                retries += 1
            elif status_code == 401:
                logging.error(f"Unauthorized (401) for API key ending with '{key_identifier}' on combining summaries. This key may be invalid or revoked.")
                return f"[Error combining summaries: API request failed - {http_err}]\n\n{combined_text_for_prompt}"
            else: # Other HTTP errors
                logging.error(f"Groq API HTTP error (combine summaries): {http_err}", exc_info=True)
                return f"[Error combining summaries: API request failed - {http_err}]\n\n{combined_text_for_prompt}"
        except requests.exceptions.RequestException as e:
            logging.error(f"Groq API request failed (combine summaries): {e}", exc_info=True)
            return f"[Error combining summaries: API request failed - {e}]\n\n{combined_text_for_prompt}"
    # If all retries fail for 429
    key_identifier = f"...{api_key_to_use[-4:]}" if len(api_key_to_use) > 4 else api_key_to_use
    logging.error(f"Max retries reached for 429 error on combining summaries with key ending '{key_identifier}'.")
    return f"[Error combining summaries: API request failed - Max retries for 429 error]\n\n{combined_text_for_prompt}"

# --- RAG-related Helper Functions ---
@st.cache_resource # Cache the model loading to avoid reloading on every script rerun
def _load_embedding_model(model_name):
    """Loads the SentenceTransformer model. Decorated with st.cache_resource."""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logging.error(f"Error loading embedding model '{model_name}': {e}", exc_info=True)
        st.error(f"Error loading embedding model '{model_name}': {e}")
        return None

def get_embedding_model():
    """Gets the embedding model, loading it if necessary using the cached function."""
    global embedding_model
    if embedding_model is None:
        with st.spinner(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'... (first-time may take a moment)"):
            embedding_model = _load_embedding_model(EMBEDDING_MODEL_NAME)
        if embedding_model:
            # Using sidebar for less intrusive notifications about model loading
            st.sidebar.success("Embedding model loaded and ready for Q&A.")
        else:
            st.sidebar.error("Embedding model failed to load. Q&A feature will be unavailable.")
            st.stop() # Stop execution if the embedding model is critical and fails to load
    return embedding_model

def chunk_text(text, chunk_size=256, chunk_overlap=32): # chunk_size in words (approx)
    # A simple word-based chunker.
    # `all-MiniLM-L6-v2` handles sequences up to 256 word pieces well.
    words = text.split() # Simple split by space
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def prepare_rag_data(transcript_text):
    """Chunks the transcript and generates embeddings for RAG. Stores in session_state."""
    model = get_embedding_model()
    if not model:
        st.error("Embedding model not available. Q&A will not function.")
        st.session_state.rag_chunks = []
        st.session_state.rag_embeddings = None
        return

    chunks = chunk_text(transcript_text)
    if not chunks:
        st.warning("No text chunks generated from the transcript for RAG.")
        st.session_state.rag_chunks = []
        st.session_state.rag_embeddings = None
        return

    with st.spinner("🧠 Generating embeddings for Q&A..."):
        st.session_state.rag_embeddings = model.encode(chunks, show_progress_bar=False)
    st.session_state.rag_chunks = chunks
    st.sidebar.info(f"RAG Ready: {len(st.session_state.rag_chunks)} text chunks prepared for Q&A.")

def retrieve_relevant_chunks(question_text):
    """Retrieves the most relevant transcript chunks for a given question from session_state."""
    if not hasattr(st.session_state, 'rag_embeddings') or \
       st.session_state.rag_embeddings is None or \
       not hasattr(st.session_state, 'rag_chunks') or \
       not st.session_state.rag_chunks:
        st.warning("RAG data not prepared. Cannot retrieve chunks.")
        return []

    model = get_embedding_model()
    if not model: return []

    question_embedding = model.encode([question_text])
    similarities = cosine_similarity(question_embedding, st.session_state.rag_embeddings)
    
    top_k_indices = similarities[0].argsort()[-RAG_TOP_K_CHUNKS:][::-1]
    relevant_chunks = [st.session_state.rag_chunks[i] for i in top_k_indices]
    return relevant_chunks

def answer_question_with_rag(api_key_to_use, question, context_chunks):
    """Uses Groq LLM to answer a question based on provided context chunks."""
    retries = 0
    time.sleep(API_CALL_DELAY_SECONDS) # Wait before making the call
    # api_key_to_use = get_next_groq_api_key() # Key is now passed in
    if not context_chunks:
        return "I couldn't find specific information in the video transcript to answer that question."

    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = (
        f"Based *only* on the following context from a video transcript, please answer the question.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    # Using the existing summarize_with_groq structure but with a different system/user prompt might be an option
    # For clarity, a separate call structure is used here.
    headers = {
        "Authorization": f"Bearer {api_key_to_use}",
        "Content-Type": "application/json",
        "User-Agent": "VideoBuddyStreamlitApp/1.0" # Good practice
    }
    body = {
        "model": GROQ_MODEL, # Or a smaller, faster model if preferred for Q&A
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based *only* on the provided text context. If the answer is not in the context, say so clearly."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2 # Lower temperature for more factual, less creative Q&A
    }
    while retries <= MAX_RETRIES_ON_429:
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=45) # Adjusted timeout
            response.raise_for_status()

            data = response.json()
            # Validate response structure
            if "choices" in data and data["choices"] and \
               isinstance(data["choices"], list) and len(data["choices"]) > 0 and \
               "message" in data["choices"][0] and \
               "content" in data["choices"][0]["message"]:
                return data["choices"][0]["message"]["content"]
            else:
                logging.error(f"Unexpected API response structure from Groq (Q&A): {data}")
                raise ValueError(f"Unexpected API response structure from Groq (Q&A). Full response: {data}") # Non-retryable
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            key_identifier = f"...{api_key_to_use[-4:]}" if len(api_key_to_use) > 4 else api_key_to_use
            if status_code == 429 and retries < MAX_RETRIES_ON_429:
                logging.warning(f"Rate limit hit (429) for API key ending with '{key_identifier}' on Q&A. Retry {retries + 1}/{MAX_RETRIES_ON_429}. Waiting {RATE_LIMIT_DELAY_SECONDS}s.")
                time.sleep(RATE_LIMIT_DELAY_SECONDS)
                retries +=1
            elif status_code == 401: # Non-retryable auth error
                logging.error(f"Unauthorized (401) for API key ending with '{key_identifier}' on Q&A. This key may be invalid or revoked.")
                raise RuntimeError(f"Failed to connect to Groq API for Q&A: {http_err}") from http_err
            else: # Other HTTP errors, non-retryable by this logic
                raise RuntimeError(f"Failed to connect to Groq API for Q&A: {http_err}") from http_err
        except requests.exceptions.RequestException as e: # Non-retryable network error
            logging.error(f"Groq API request failed (Q&A): {e}", exc_info=True)
            raise RuntimeError(f"Failed to connect to Groq API for Q&A: {e}") from e
    # If all retries fail for 429
    key_identifier = f"...{api_key_to_use[-4:]}" if len(api_key_to_use) > 4 else api_key_to_use
    logging.error(f"Max retries reached for 429 error on Q&A with key ending '{key_identifier}'.")
    raise RuntimeError(f"Failed to connect to Groq API for Q&A: Max retries for 429 error")

# --- Streamlit UI ---
st.set_page_config(
    page_title="🎥 Video Buddy",
    page_icon="📝", # Changed icon slightly
    layout="wide", # Changed layout
    initial_sidebar_state="expanded"
)

# ---------- Custom Styling (from video_buddy_ui.py) ----------
st.markdown("""
    <style>
        .main {
            /* background-color: #fff; */ /* Let Streamlit handle main bg for theme compatibility */
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 { /* Applied to all header levels for consistency */
            text-align: center;
            color: #FF6F61; /* Primary color for headers */
        }
        .stButton>button {
            background-color: #FF6F61;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 24px;
            border: none; /* Remove default border */
            display: block; /* Make button take full width if needed or center it */
            margin-left: auto; /* Center the button */
            margin-right: auto; /* Center the button */
        }
        .stButton>button:hover {
            background-color: #E55A50; /* Darker shade on hover */
            color: white;
        }
        .stTextInput>div>input, .stTextArea>div>textarea { /* Apply to text_area too if used */
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #ddd; /* Softer border */
        }
        footer {visibility: hidden;} /* Hides default Streamlit footer */
    </style>
""", unsafe_allow_html=True)

if not AVAILABLE_GROQ_API_KEYS: # Check if any API keys are available
    st.error("🔴 Critical Error: No GROQ API keys are configured. Please set the GROQ_API_KEYS environment variable (comma-separated) or ensure fallback keys are present in the script. The application cannot function without at least one API key.")
    st.stop()

# --- Header Section ---
st.markdown("""
    <h1 style='text-align: center; color: #FF6F61;'>🎬 Video Buddy</h1>
    <h5 style='text-align: center; color: #FF6F61; margin-bottom: 2rem;'><em>Turn your YouTube videos into fun, easy-to-read notes!</em> 📝</h5>
    """, unsafe_allow_html=True)

# st.markdown("---") # Optional separator, can be removed if layout is clean

# --- Input Section ---
st.markdown("<h4 style='text-align: center; color: #FF6F61;'>🔗 Paste your YouTube Link below:</h4>", unsafe_allow_html=True)
video_url = st.text_input(
    "", # No label above the input field itself
    placeholder="e.g. https://www.youtube.com/watch?v=abc123",
    label_visibility="collapsed" # Hides the auto-generated label
)

add_vertical_space(2) # More space before the button

# --- Options Section (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Options")
    st.write("Fine-tune transcription quality:") # From video_buddy_ui.py
    selected_whisper_model_display = st.selectbox(
        "Transcription Model:", # Changed label slightly
        list(WHISPER_MODEL_OPTIONS_DISPLAY.keys()),
        index=2, # Default to "base.en"
        help="Smaller models are faster but may be less accurate. 'base.en' offers a good balance. Higher quality may take longer to transcribe."
    )
    st.info("Higher quality models may take longer to transcribe.") # From video_buddy_ui.py

# Initialize session state for RAG if not already present
if 'rag_chunks' not in st.session_state:
    st.session_state.rag_chunks = []
if 'rag_embeddings' not in st.session_state:
    st.session_state.rag_embeddings = None
# Initialize session state for summary and transcript
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None


if st.button("✨ Make My Notes!"):
    if not video_url or video_url.strip() == "": # Check if empty or just whitespace
        st.warning("Oops! You forgot to paste a video link 😅")
        st.stop()

    # --- URL Pre-validation ---
    # This is a basic sanity check. yt-dlp will do the definitive validation.
    # Goal: Catch obviously incorrect inputs like "HIII" or incomplete URLs quickly.
    temp_url_for_check = video_url
    # If scheme is missing, and it looks like it might be a YouTube URL, prepend https for parsing
    if not video_url.startswith(('http://', 'https://')) and \
       ("youtube.com" in video_url or "youtu.be" in video_url):
        temp_url_for_check = 'https://' + video_url

    try:
        parsed = urlparse(temp_url_for_check)
        is_valid_scheme = parsed.scheme in ['http', 'https']
        is_youtube_domain = parsed.netloc.endswith('youtube.com') or \
                            parsed.netloc == 'youtu.be'

        has_meaningful_path = False
        if is_youtube_domain:
            if parsed.netloc == 'youtu.be':
                # For youtu.be, path should be like /VIDEOID
                has_meaningful_path = bool(parsed.path and len(parsed.path) > 1 and parsed.path != "/")
            elif parsed.netloc.endswith('youtube.com'):
                # For youtube.com, check for common video/shorts/live paths
                has_meaningful_path = (parsed.path == '/watch' or \
                                       parsed.path.startswith('/shorts/') or \
                                       parsed.path.startswith('/live/') or \
                                       parsed.path.startswith('/embed/'))

        if not (is_valid_scheme and is_youtube_domain and has_meaningful_path):
            st.error(f"❌ Invalid or incomplete YouTube URL: '{video_url}'. Please use a full URL to a specific video, shorts, or live stream (e.g., https://www.youtube.com/watch?v=VIDEO_ID).")
            st.stop()
            
    except ValueError: # urlparse can raise ValueError for some malformed URLs
        st.error(f"❌ Invalid URL format: '{video_url}'. Please enter a valid YouTube video URL.")
        st.stop()

    # Determine actual model name from user's display selection
    actual_whisper_model_to_use = WHISPER_MODEL_OPTIONS_DISPLAY.get(selected_whisper_model_display, DEFAULT_WHISPER_MODEL_SIZE)
    # If pre-validation passes, proceed with processing
    # Instead of a generic spinner, we'll use st.progress for more granular feedback

    start_time = time.time() # Record start time
    progress_bar = st.progress(0, text="🚀 Starting process...")
    # Reset RAG data for new video processing
    st.session_state.rag_chunks = []
    st.session_state.rag_embeddings = None
    st.session_state.summary = None # Reset summary for new video
    st.session_state.transcript = None # Reset transcript for new video
    
    session_api_key = None # Key to be used for this entire "Make My Notes!" session
    session_key_tried_indices = set() # Keep track of keys tried in this session to avoid infinite loops with bad keys

    def update_progress_with_timer(progress_value, text_message):
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        progress_bar.progress(progress_value, text=f"{text_message} (Elapsed: {minutes:02d}:{seconds:02d})")

    try:
        # --- Robust Session API Key Selection ---
        initial_key_index = CURRENT_API_KEY_INDEX
        session_api_key = AVAILABLE_GROQ_API_KEYS[initial_key_index]
        session_key_tried_indices.add(initial_key_index)
        logging.info(f"Attempting to use API key ending '...{session_api_key[-4:]}' (index {initial_key_index}) for this session.")

        # Helper function to try getting a new key if the current one fails with 401
        def get_new_session_key_on_401(current_failed_key_index):
            global CURRENT_API_KEY_INDEX
            logging.warning(f"API Key at index {current_failed_key_index} (ends '...{AVAILABLE_GROQ_API_KEYS[current_failed_key_index][-4:]}') failed with 401. Trying next key for this session.")
            
            # Try to find the next available key that hasn't been tried this session
            for i in range(1, len(AVAILABLE_GROQ_API_KEYS)):
                next_try_index = (current_failed_key_index + i) % len(AVAILABLE_GROQ_API_KEYS)
                if next_try_index not in session_key_tried_indices:
                    session_key_tried_indices.add(next_try_index)
                    new_key = AVAILABLE_GROQ_API_KEYS[next_try_index]
                    logging.info(f"Switched to API key ending '...{new_key[-4:]}' (index {next_try_index}) for this session after previous 401.")
                    # Update the global CURRENT_API_KEY_INDEX to start from here next time the app is fully used
                    CURRENT_API_KEY_INDEX = next_try_index 
                    return new_key
            logging.error("All available API keys have been tried and failed with 401 this session.")
            return None

        # Advance the global key index for the *next* fresh "Make My Notes!" session.
        # This ensures that even if this session sticks to one key, the next overall app use tries a different starting key.
        CURRENT_API_KEY_INDEX = (CURRENT_API_KEY_INDEX + 1) % len(AVAILABLE_GROQ_API_KEYS) # Advance for next time

        update_progress_with_timer(10, "⬇️ Downloading audio from YouTube...")
        downloaded_audio_path = download_youtube_audio(video_url) # This is the original video_url from user input

        with TempAudioFile(downloaded_audio_path) as audio_path: # Ensures cleanup
            update_progress_with_timer(20, "🎤 Transcribing audio... (this can take a while)")
            transcript = transcribe_with_local_whisper(audio_path, model_size=actual_whisper_model_to_use)
            
            st.session_state.transcript = transcript 

            # --- Chunked Summarization ---
            if not transcript.strip():
                st.warning("Transcription resulted in empty text. Cannot summarize.")
                summary = "No summary available as the transcription was empty."
            else:
                transcript_chunks = split_text_into_chunks(transcript, MAX_CHARS_PER_SUMMARY_CHUNK, SUMMARY_CHUNK_OVERLAP_CHARS)
                if not transcript_chunks:
                    st.warning("Transcript could not be split into chunks. Using full transcript for summary (might be slow or fail for very long videos).")
                     # Fallback to a direct (potentially problematic) summarization if chunking fails badly
                    # This part needs a robust single-chunk summarizer if we want a fallback.
                    # For now, let's assume chunking works or we report an issue.
                    summary = "Error: Could not process transcript for summarization."
                else:
                    all_chunk_summaries = []
                    num_chunks = len(transcript_chunks)
                    for i, chunk in enumerate(transcript_chunks):
                        update_progress_with_timer(50 + int((i / num_chunks) * 40), f"✍️ Summarizing part {i+1}/{num_chunks}...")
                        if not session_api_key: # Should not happen if logic is correct, but as a safeguard
                            raise RuntimeError("No valid API key available for summarization.")
                        
                        chunk_summary = summarize_transcript_chunk_with_groq(session_api_key, chunk, i + 1, num_chunks)
                        
                        # Handle 401 on the first chunk by trying to switch key for the session
                        if i == 0 and chunk_summary.startswith("[Error summarizing part") and "401 Client Error" in chunk_summary:
                            original_failed_key_index = AVAILABLE_GROQ_API_KEYS.index(session_api_key) # Find index of the failed key
                            new_key_for_session = get_new_session_key_on_401(original_failed_key_index)
                            if new_key_for_session:
                                session_api_key = new_key_for_session # Switch key for the rest of the session
                                chunk_summary = summarize_transcript_chunk_with_groq(session_api_key, chunk, i + 1, num_chunks) # Retry the first chunk with new key
                            # If new_key_for_session is None, all keys failed, error will propagate
                        all_chunk_summaries.append(chunk_summary)
                    
                    update_progress_with_timer(90, "📚 Combining all notes...")
                    summary = combine_summaries_with_groq(session_api_key, all_chunk_summaries, original_video_url=video_url)

            # Store results in session state
            st.session_state.summary = summary
            update_progress_with_timer(100, "✅ All done!")
            st.success("✅ All done!")
            # --- Prepare RAG data after successful summary generation ---
            prepare_rag_data(st.session_state.transcript) # Use the full transcript for RAG
    except Exception as e:
        current_key_for_error_msg = session_api_key if session_api_key else "Unknown"
        key_identifier_for_error = f"...{current_key_for_error_msg[-4:]}" if len(current_key_for_error_msg) > 4 else current_key_for_error_msg
        
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
            logging.error(f"An API key (potentially ending '...{key_identifier_for_error}') resulted in an Unauthorized (401) error during processing. This key may be invalid.")
            st.error(f"😢 Something went wrong: The API key used for this session is unauthorized. Please check your API keys. Error: {e}")
        else:
            st.error(f"😢 Something went wrong: {e}")
        logging.exception("An error occurred during the main 'Make My Notes!' process:") # Logs full traceback
    finally:
        # Ensure the progress bar is removed if it's still visible after success or if an error occurred before its removal
        if progress_bar is not None: # progress_bar is always defined before this try/finally block
             progress_bar.empty() # Clears the progress bar from the UI

# --- Display Summary and Transcript (if available in session state) ---
if st.session_state.summary:
    st.markdown("<h3 style='text-align: center; color: #FF6F61; margin-top: 2rem;'>📝 Your Easy Notes:</h3>", unsafe_allow_html=True)
    # Use columns for better layout of summary and download button
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(st.session_state.summary) # Use markdown for potentially better formatting if summary contains markdown
    with col2:
        st.download_button("📥 Download Notes", st.session_state.summary, file_name="video_buddy_notes.txt", mime="text/plain", use_container_width=True)

if st.session_state.transcript:
    add_vertical_space(1)
    with st.expander("📜 What I Heard (Transcript Preview)", expanded=False):
        st.text_area("", st.session_state.transcript, height=200, disabled=True, label_visibility="collapsed")

# --- Q&A Section (Appears after summary is generated and RAG data is ready) ---
if st.session_state.rag_chunks: # Only show Q&A if RAG data is ready
    add_vertical_space(2)
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #FF6F61;'>🤔 Ask a Question About the Video:</h3>", unsafe_allow_html=True)
    user_question = st.text_input("Your question:", key="user_qna_question", placeholder="Ask something about the video content...")

    if user_question:
        with st.spinner("🔍 Searching for answers in the video..."):
            relevant_chunks = retrieve_relevant_chunks(user_question)
            if relevant_chunks:
                # For Q&A, we should also use the session_api_key if available from a successful run
                # However, session_api_key is local to the button click.
                # A more robust way would be to store the last successful key in st.session_state
                if 'session_api_key' in st.session_state and st.session_state.session_api_key:
                    qna_api_key = st.session_state.session_api_key
                else: # Fallback if no session key was stored (e.g. main process failed before setting it)
                    qna_api_key = get_initial_groq_api_key() 
                
                answer = answer_question_with_rag(qna_api_key, user_question, relevant_chunks)
                st.markdown("<h4 style='text-align: center; color: #FF6F61;'>💡 Answer:</h4>", unsafe_allow_html=True)
                st.info(answer) # Using st.info for a slightly different visual style for answers
            else:
                st.warning("Could not find relevant information in the transcript to answer your question confidently.")

# --- Footer Image or Fun Element ---
add_vertical_space(2)
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray; font-size:small; margin-top: 2rem;'>Video Buddy — Happy Note-Making! 🎉</div>", unsafe_allow_html=True)

# It's good practice to have a requirements.txt file for your project.
# Example requirements.txt content:
# streamlit
# faster-whisper
# sentence-transformers
# scikit-learn
# requests
# yt-dlp # Ensure yt-dlp is installed, or provide instructions for users
# streamlit-extras
