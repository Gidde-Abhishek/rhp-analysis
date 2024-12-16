import os
import pdfplumber
from llama_cpp import Llama

#################################
# Configuration
#################################
PDF_PATH = "mobikwik.pdf"
MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_M.gguf"

PAGE_SUMMARY_WORD_LIMIT = 150
SECTION_SUMMARY_WORD_LIMIT = 300
CHUNK_SIZE = 20

LLAMA_PARAMS = {
    "model_path": MODEL_PATH,
    "n_ctx": 4096,          # simpler context window
    "n_threads": 16,        # reasonable CPU threads
    "n_gpu_layers": 40,     # offload layers to GPU if supported
    "f16_kv": True,
    "temperature": 0.7,     # increase temperature for more elaboration
    "top_p": 0.9,
    "max_tokens": 2048,     # limit generation length, can adjust if needed
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

Now produce the final IPO note below. Be detailed and do not stop early. Begin now:
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
    response = client(
        prompt=prompt,
        echo=False,
        # No stop tokens here
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
    print("Loading LLaMA model...")
    client = Llama(**LLAMA_PARAMS)

    pages_text = extract_pages(PDF_PATH)
    print(f"Extracted {len(pages_text)} pages from {PDF_PATH}")

    page_summaries = []
    for i, p_text in enumerate(pages_text):
        print(f"Summarizing page {i+1}/{len(pages_text)}...")
        summary = summarize_page(client, p_text, word_limit=PAGE_SUMMARY_WORD_LIMIT)
        page_summaries.append(summary)

    section_summaries = []
    for i in range(0, len(page_summaries), CHUNK_SIZE):
        chunk = page_summaries[i:i+CHUNK_SIZE]
        chunk_joined = "\n\n".join(chunk)
        start_page = i+1
        end_page = i+len(chunk)
        print(f"Summarizing section: pages {start_page} to {end_page}")
        section_summary = summarize_section(client, chunk_joined, word_limit=SECTION_SUMMARY_WORD_LIMIT)
        section_summaries.append(section_summary)

    all_section_summaries = "\n\n".join(section_summaries)
    print("Assembling final IPO note...")
    final_ipo_note = assemble_final_note(client, all_section_summaries)

    # Save to file for inspection
    with open("final_ipo_note.txt", "w") as f:
        f.write(final_ipo_note)

    print("Final IPO Note:\n")
    print(final_ipo_note)

if __name__ == "__main__":
    main()
