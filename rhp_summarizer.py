# import os
# import pdfplumber
# from llama_cpp import Llama

# #################################
# # Configuration
# #################################
# PDF_PATH = "mobikwik.pdf"
# MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_M.gguf"

# PAGE_SUMMARY_WORD_LIMIT = 150
# SECTION_SUMMARY_WORD_LIMIT = 300
# CHUNK_SIZE = 20

# LLAMA_PARAMS = {
#     "model_path": MODEL_PATH,
#     "n_ctx": 15360,
#     "n_threads": 32,
#     "n_gpu_layers": -1,
#     "f16_kv": True,
#     "temperature": 0.7,
#     "top_p": 0.9,
#     "max_tokens": 2048,
#     "verbose": False
# }

# #################################
# # Prompts
# #################################
# PAGE_SUMMARY_PROMPT_TEMPLATE = """You are an AI assistant with expertise in analyzing financial documents.

# Given the following page from a Red Herring Prospectus, produce a concise summary (no more than {word_limit} words) highlighting:
# - Relevant company details (business lines, market position, strategy)
# - Financial data points (revenues, profitability) if mentioned
# - Risk factors if mentioned
# - Any key IPO details (if present)

# Keep it factual and concise.

# Page Content:
# {page_text}

# Your Summary:"""

# SECTION_SUMMARY_PROMPT_TEMPLATE = """You have the following summaries of multiple pages from an RHP:
# {summaries}

# Please produce a higher-level summary (no more than {word_limit} words) integrating all these details. Focus on key financials, strategic insights, risk factors, and IPO-related info that appear repeatedly or stand out as important. Provide a coherent narrative that merges the information from these pages.

# Your Integrated Summary:"""

# FINAL_NOTE_PROMPT_TEMPLATE = """You have been provided integrated summaries of multiple sections of a Red Herring Prospectus. Use them to produce a concise, structured IPO note that includes:

# - Cover Page info (Company name, tagline)
# - Company description (main business lines, position)
# - Issue Details (tables for price band, face value, issue size, etc.)
# - Salient Features (company overview, IPO proceeds usage, strengths, risk factors, market opportunity)
# - Financial Analysis (tables and key metrics)
# - Peer Comparison (table of key metrics across competitors)
# - Detailed Financials (tables: P&L, Balance Sheet, Cash Flows, key ratios)
# - IPO Timeline (key dates)
# - Investment Recommendation, Rationale, Risk factors
# - Legal Disclaimers

# Use a professional, structured format. Present tables in Markdown format. Ensure clarity and coherence.

# Section Summaries:
# {all_section_summaries}

# Now produce the final IPO note below. Be detailed and do not stop early. Begin now:
# """

# #################################
# # Functions
# #################################

# def extract_pages(pdf_path):
#     pages_text = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text() or ""
#             pages_text.append(text)
#     return pages_text

# def run_local_llm(client: Llama, prompt: str, max_tokens=None) -> str:
#     # If max_tokens is not provided, it will default to the model's setting.
#     response = client(
#         prompt=prompt,
#         echo=False,
#         max_tokens=max_tokens if max_tokens else None
#     )
#     return response['choices'][0]['text'].strip()

# def summarize_page(client: Llama, page_text, word_limit=150):
#     prompt = PAGE_SUMMARY_PROMPT_TEMPLATE.format(page_text=page_text, word_limit=word_limit)
#     return run_local_llm(client, prompt)

# def summarize_section(client: Llama, summaries_text, word_limit=300):
#     prompt = SECTION_SUMMARY_PROMPT_TEMPLATE.format(summaries=summaries_text, word_limit=word_limit)
#     return run_local_llm(client, prompt)

# def assemble_final_note(client: Llama, section_summaries_text, max_tokens=4096):
#     prompt = FINAL_NOTE_PROMPT_TEMPLATE.format(all_section_summaries=section_summaries_text)
#     return run_local_llm(client, prompt, max_tokens=max_tokens)

# #################################
# # Main Pipeline
# #################################
# def main():
#     print("Loading LLaMA model...")
#     client = Llama(**LLAMA_PARAMS)

#     pages_text = extract_pages(PDF_PATH)
#     print(f"Extracted {len(pages_text)} pages from {PDF_PATH}")

#     # Ensure log directory
#     os.makedirs("logs", exist_ok=True)

#     page_summaries = []
#     for i, p_text in enumerate(pages_text):
#         print(f"Summarizing page {i+1}/{len(pages_text)}...")
#         summary = summarize_page(client, p_text, word_limit=PAGE_SUMMARY_WORD_LIMIT)
#         page_summaries.append(summary)

#         # Log page summary
#         with open(f"logs/page_{i+1}_summary.txt", "w") as f:
#             f.write(summary)

#     section_summaries = []
#     for i in range(0, len(page_summaries), CHUNK_SIZE):
#         chunk = page_summaries[i:i+CHUNK_SIZE]
#         chunk_joined = "\n\n".join(chunk)
#         start_page = i+1
#         end_page = i+len(chunk)
#         print(f"Summarizing section: pages {start_page} to {end_page}")
#         section_summary = summarize_section(client, chunk_joined, word_limit=SECTION_SUMMARY_WORD_LIMIT)
#         section_summaries.append(section_summary)

#         # Log section summary
#         with open(f"logs/section_{start_page}_to_{end_page}_summary.txt", "w") as f:
#             f.write(section_summary)

#     all_section_summaries = "\n\n".join(section_summaries)
#     print("Assembling final IPO note...")

#     # Log final prompt before generation
#     final_prompt = FINAL_NOTE_PROMPT_TEMPLATE.format(all_section_summaries=all_section_summaries)
#     with open("logs/final_prompt.txt", "w") as f:
#         f.write(final_prompt)

#     final_ipo_note = assemble_final_note(client, all_section_summaries, max_tokens=8192)

#     # Save final output to file
#     with open("final_ipo_note.txt", "w") as f:
#         f.write(final_ipo_note)

#     print("Final IPO Note:\n")
#     print(final_ipo_note)

# if __name__ == "__main__":
#     main()

# NEW FLOW

import os
import pdfplumber
from llama_cpp import Llama
import re
import tiktoken
from typing import List, Tuple
import math

#################################
# Configuration
#################################
PDF_PATH = "mobikwik.pdf"
MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_M.gguf"

# Adjusted limits for token window
PAGE_SUMMARY_WORD_LIMIT = 100      # Reduced from 150
SECTION_SUMMARY_WORD_LIMIT = 200   # Reduced from 300
CHUNK_SIZE = 15                    # Reduced from 20

# Pages to skip (case-insensitive patterns)
SKIP_PAGE_PATTERNS = [
    r"legal\s+disclaimer",
    r"risk\s+factors",
    r"definitions?\s+and\s+abbreviations?",
    r"general\s+information",
    r"notice\s+to\s+investors",
    r"declaration",
    r"presentation\s+of\s+financial",
    r"industry\s+overview",
    r"basis\s+for\s+issue\s+price",
]

LLAMA_PARAMS = {
    "model_path": MODEL_PATH,
    "n_ctx": 15360,
    "n_threads": 32,
    "n_gpu_layers": -1,
    "n_batch": 1512,           # Increased for better GPU utilization
    "f16_kv": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "verbose": False,
    "gpu_memory_utilization": 0.95
}

#################################
# Prompts
#################################
PAGE_SUMMARY_PROMPT_TEMPLATE = """You are an AI assistant with expertise in analyzing financial documents.

Given the following page from a Red Herring Prospectus, produce a concise summary (no more than {word_limit} words) highlighting:
- Relevant company details (business lines, market position, strategy)
- Financial data points (revenues, profitability) if mentioned
- Risk factors if mentioned
- Any key IPO details (if present)

Keep it factual and concise.

Page Content:
{page_text}

Your Summary:"""

SECTION_SUMMARY_PROMPT_TEMPLATE = """You have the following summaries of multiple pages from an RHP:
{summaries}

Please produce a higher-level summary (no more than {word_limit} words) integrating all these details. Focus on key financials, strategic insights, risk factors, and IPO-related info that appear repeatedly or stand out as important. Provide a coherent narrative that merges the information from these pages.

Your Integrated Summary:"""

FINAL_NOTE_PROMPT_TEMPLATE = """You have been provided integrated summaries of multiple sections of a Red Herring Prospectus. Use them to produce a concise, structured IPO note that includes:

- Cover Page info (Company name, tagline)
- Company description (main business lines, position)
- Issue Details (tables for price band, face value, issue size, etc.)
- Salient Features (company overview, IPO proceeds usage, strengths, risk factors, market opportunity)
- Financial Analysis (tables and key metrics)
- Peer Comparison (table of key metrics across competitors)
- Detailed Financials (tables: P&L, Balance Sheet, Cash Flows, key ratios)
- IPO Timeline (key dates)
- Investment Recommendation, Rationale, Risk factors
- Legal Disclaimers

Use a professional, structured format. Present tables in Markdown format. Ensure clarity and coherence.

Section Summaries:
{all_section_summaries}

Now produce the final IPO note below. Be detailed and do not stop early. Begin now:"""

#################################
# Functions
#################################
def should_skip_page(text):
    """Check if page should be skipped based on patterns."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in SKIP_PAGE_PATTERNS)

def clean_text(text):
    """Clean and truncate text to prevent token overflow."""
    # Remove multiple spaces, newlines, and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:()\[\]{}\-"\'₹$%]', '', text)
    # Truncate to approximate token limit (assuming ~4 chars per token)
    max_chars = 3000  # (~750 tokens)
    return text[:max_chars]

def extract_pages(pdf_path):
    pages_text = []
    page_numbers = []  # Track original page numbers
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not should_skip_page(text):
                pages_text.append(clean_text(text))
                page_numbers.append(i + 1)
    
    return pages_text, page_numbers

def run_local_llm(client: Llama, prompt: str, max_tokens=None) -> str:
    response = client(
        prompt=prompt,
        echo=False,
        max_tokens=max_tokens if max_tokens else None
    )
    return response['choices'][0]['text'].strip()

def summarize_page(client: Llama, page_text, word_limit=100):
    prompt = PAGE_SUMMARY_PROMPT_TEMPLATE.format(page_text=page_text, word_limit=word_limit)
    return run_local_llm(client, prompt)

def summarize_section(client: Llama, summaries_text, word_limit=200):
    prompt = SECTION_SUMMARY_PROMPT_TEMPLATE.format(summaries=summaries_text, word_limit=word_limit)
    return run_local_llm(client, prompt)

def assemble_final_note(client: Llama, section_summaries_text, max_tokens=4096):
    prompt = FINAL_NOTE_PROMPT_TEMPLATE.format(all_section_summaries=section_summaries_text)
    return run_local_llm(client, prompt, max_tokens=max_tokens)

def process_pages_in_batches(client, pages_text, page_numbers, batch_size=5):
    """Process pages in small batches to prevent memory issues."""
    summaries = []
    
    for i in range(0, len(pages_text), batch_size):
        batch_texts = pages_text[i:i+batch_size]
        batch_numbers = page_numbers[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{len(pages_text)//batch_size + 1}")
        
        for j, (text, page_num) in enumerate(zip(batch_texts, batch_numbers)):
            try:
                print(f"  Summarizing page {page_num}")
                summary = summarize_page(client, text, word_limit=PAGE_SUMMARY_WORD_LIMIT)
                summaries.append((page_num, summary))
                
                # Log individual summaries
                with open(f"logs/page_{page_num}_summary.txt", "w") as f:
                    f.write(f"Page {page_num}:\n{summary}")
                    
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                continue
    
    return summaries

#################################
# Main Pipeline
#################################
def main():
    print("Loading LLaMA model...")
    client = Llama(**LLAMA_PARAMS)

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Extract and filter pages
    pages_text, page_numbers = extract_pages(PDF_PATH)
    print(f"Extracted {len(pages_text)} relevant pages from {PDF_PATH}")

    # Process pages in batches
    page_summaries = process_pages_in_batches(client, pages_text, page_numbers)
    
    # Sort summaries by page number
    page_summaries.sort(key=lambda x: x[0])
    
    # Group summaries into sections
    section_summaries = []
    for i in range(0, len(page_summaries), CHUNK_SIZE):
        chunk = page_summaries[i:i+CHUNK_SIZE]
        chunk_text = "\n\n".join([f"Page {num}: {summary}" for num, summary in chunk])
        
        start_page = chunk[0][0]
        end_page = chunk[-1][0]
        print(f"Summarizing section: pages {start_page} to {end_page}")
        
        section_summary = summarize_section(client, chunk_text, word_limit=SECTION_SUMMARY_WORD_LIMIT)
        section_summaries.append(section_summary)
        
        with open(f"logs/section_{start_page}_to_{end_page}_summary.txt", "w") as f:
            f.write(section_summary)

    # Assemble final note with truncated input
    all_section_summaries = "\n\n".join(section_summaries)
    print("Assembling final IPO note...")
    
    final_ipo_note = assemble_final_note(client, all_section_summaries)
    
    with open("final_ipo_note.txt", "w") as f:
        f.write(final_ipo_note)

    print("Final IPO Note saved to final_ipo_note.txt")

def count_tokens(text: str) -> int:
    """Approximate token count using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_summaries_by_tokens(summaries: List[Tuple[int, str]], target_tokens: int = 10000) -> List[List[Tuple[int, str]]]:
    """Split summaries into chunks that fit within token limit."""
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for page_num, summary in summaries:
        summary_tokens = count_tokens(summary) + 100  # Add buffer for formatting
        
        if current_tokens + summary_tokens > target_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append((page_num, summary))
        current_tokens += summary_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Modify the main() function to use token-based chunking
def main():
    print("Loading LLaMA model...")
    client = Llama(**LLAMA_PARAMS)

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Extract and filter pages
    pages_text, page_numbers = extract_pages(PDF_PATH)
    print(f"Extracted {len(pages_text)} relevant pages from {PDF_PATH}")

    # Process pages in batches
    page_summaries = process_pages_in_batches(client, pages_text, page_numbers)
    
    # Sort summaries by page number
    page_summaries.sort(key=lambda x: x[0])
    
    # Group summaries into token-sized chunks
    chunks = chunk_summaries_by_tokens(page_summaries, target_tokens=7000)  # Conservative limit
    
    # Process each chunk into section summaries
    section_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_text = "\n\n".join([f"Page {num}: {summary}" for num, summary in chunk])
        start_page = chunk[0][0]
        end_page = chunk[-1][0]
        
        print(f"Summarizing section {i+1}/{len(chunks)}: pages {start_page} to {end_page}")
        try:
            section_summary = summarize_section(client, chunk_text, word_limit=SECTION_SUMMARY_WORD_LIMIT)
            section_summaries.append(section_summary)
            
            with open(f"logs/section_{start_page}_to_{end_page}_summary.txt", "w") as f:
                f.write(section_summary)
        except Exception as e:
            print(f"Error processing section {i+1}: {str(e)}")
            continue

    # Combine section summaries with token limit in mind
    all_summaries_text = "\n\n".join(section_summaries)
    if count_tokens(all_summaries_text) > 10000:  # Conservative limit for final summary
        # Truncate or further summarize if needed
        section_summaries = section_summaries[:math.floor(10000/count_tokens(section_summaries[0]))]
        all_summaries_text = "\n\n".join(section_summaries)
    
    print("Assembling final IPO note...")
    try:
        final_ipo_note = assemble_final_note(client, all_summaries_text)
        with open("final_ipo_note.txt", "w") as f:
            f.write(final_ipo_note)
        print("Final IPO Note saved to final_ipo_note.txt")
    except Exception as e:
        print(f"Error in final note generation: {str(e)}")
        # Attempt to save what we have
        with open("section_summaries.txt", "w") as f:
            f.write(all_summaries_text)
        print("Section summaries saved to section_summaries.txt")

if __name__ == "__main__":
    main()