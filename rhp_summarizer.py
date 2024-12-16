import os
import pdfplumber
from llama_cpp import Llama

#################################
# Configuration
#################################
PDF_PATH = "mobikwik.pdf"  # Update with your RHP PDF filename
MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_M.gguf"  # Path to your GGML model file

PAGE_SUMMARY_WORD_LIMIT = 150
SECTION_SUMMARY_WORD_LIMIT = 300
CHUNK_SIZE = 20  # Number of page summaries per intermediate summarization step

# Adjust LLaMA parameters if needed.
LLAMA_PARAMS = {
    "model_path": MODEL_PATH,
    "n_ctx": 8192,
    "n_threads": 32,
    "n_gpu_layers": -1,   # All layers on GPU if supported
    "n_batch": 2048,
    "f16_kv": True,
    "f16": True,
    "use_mlock": True,
    "use_mmap": True,
    "offload_kqv": True,
    "mul_mat_q": True,
    "tensor_split": None,
    "rope_scaling": {"type": "linear", "factor": 4.0},
    "numa": True,
    "gpu_memory_utilization": 0.9,
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 8192,
    "embedding": False,  # Embedding not needed here
    "verbose": False
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

Now produce the final IPO note below. Do not stop early:
"""

#################################
# Functions
#################################

def extract_pages(pdf_path):
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return pages_text

def run_local_llm(client: Llama, prompt: str) -> str:
    # Removed stop tokens to avoid premature cutoff.
    response = client(
        prompt=prompt,
        echo=False,
        temperature=0.1,
        top_p=0.9,
        # Allow the model to generate as many tokens as needed (up to the max_tokens in LLAMA_PARAMS).
    )
    return response['choices'][0]['text'].strip()

def summarize_page(client: Llama, page_text, word_limit=150):
    prompt = PAGE_SUMMARY_PROMPT_TEMPLATE.format(page_text=page_text, word_limit=word_limit)
    return run_local_llm(client, prompt)

def summarize_section(client: Llama, summaries_text, word_limit=300):
    prompt = SECTION_SUMMARY_PROMPT_TEMPLATE.format(summaries=summaries_text, word_limit=word_limit)
    return run_local_llm(client, prompt)

def assemble_final_note(client: Llama, section_summaries_text):
    prompt = FINAL_NOTE_PROMPT_TEMPLATE.format(all_section_summaries=section_summaries_text)
    return run_local_llm(client, prompt)

#################################
# Main Pipeline
#################################
def main():
    # Load the LLaMA model
    print("Loading LLaMA model, this may take a while...")
    client = Llama(**LLAMA_PARAMS)

    # 1. Extract all pages
    pages_text = extract_pages(PDF_PATH)
    print(f"Extracted {len(pages_text)} pages from {PDF_PATH}")

    # 2. Page-by-page summaries
    page_summaries = []
    for i, p_text in enumerate(pages_text):
        print(f"Summarizing page {i+1}/{len(pages_text)}...")
        summary = summarize_page(client, p_text, word_limit=PAGE_SUMMARY_WORD_LIMIT)
        page_summaries.append(summary)

    # 3. Chunk the page summaries into sections
    section_summaries = []
    for i in range(0, len(page_summaries), CHUNK_SIZE):
        chunk = page_summaries[i:i+CHUNK_SIZE]
        chunk_joined = "\n\n".join(chunk)
        start_page = i+1
        end_page = i+len(chunk)
        print(f"Summarizing section: pages {start_page} to {end_page}")
        section_summary = summarize_section(client, chunk_joined, word_limit=SECTION_SUMMARY_WORD_LIMIT)
        section_summaries.append(section_summary)

    # 4. Final assembly of the entire IPO note
    all_section_summaries = "\n\n".join(section_summaries)
    print("Assembling final IPO note...")
    final_ipo_note = assemble_final_note(client, all_section_summaries)

    # 5. Output final note
    print("Final IPO Note:\n")
    print(final_ipo_note)

if __name__ == "__main__":
    main()
