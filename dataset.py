import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-3-opus"


def call_claude_opus(prompt: str, max_tokens: int = 500) -> str:
    """Call Claude Opus via OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/llm-infra-bench",
        "X-Title": "LLM Infra Benchmark Dataset Generator"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.8
    }
    
    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]


def generate_diverse_prompts() -> list[dict]:
    """Generate diverse benchmark prompts using Claude Opus."""
    
    generation_prompt = """Generate exactly 18 diverse LLM benchmark prompts for testing inference infrastructure. 
    
Each prompt should be unique and fall into one of these categories:
1. Coding tasks (Python, JavaScript, algorithms)
2. Reasoning/Logic puzzles
3. Creative writing (stories, poems, haikus)
4. Summarization tasks
5. JSON/structured data extraction
6. Math problems
7. Question answering
8. Translation tasks
9. Code debugging

For each prompt, provide:
- The prompt text itself
- A suggested max_output_tokens (between 50-512 based on task complexity)
- The category name

Format your response as a valid JSON array with objects containing: "prompt", "max_output_tokens", "category"

Example format:
[
  {"prompt": "Write a Python function that...", "max_output_tokens": 256, "category": "coding"},
  {"prompt": "Solve this logic puzzle...", "max_output_tokens": 150, "category": "reasoning"}
]

Generate exactly 18 diverse prompts covering all categories. Return ONLY the JSON array, no other text."""

    print("Calling Claude Opus via OpenRouter to generate prompts...")
    response = call_claude_opus(generation_prompt, max_tokens=4000)
    
    # Parse the JSON response
    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            prompts = json.loads(json_str)
        else:
            prompts = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response[:500]}...")
        raise
    
    return prompts


def create_dataset():
    """Create the benchmark dataset using Claude Opus generated prompts."""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set. "
                        "Please set it in your .env file or environment.")
    
    dataset = []

    print("=" * 60)
    print("Generating Synthetic Dataset using Claude Opus (OpenRouter)")
    print("=" * 60)
    
    generated_prompts = generate_diverse_prompts()
    
    print(f"\nReceived {len(generated_prompts)} prompts from Claude Opus")
    
    # Limit to 15-20 prompts as requested
    prompts_to_use = generated_prompts[:18]
    
    for i, item in enumerate(prompts_to_use):
        prompt_text = item.get("prompt", "")
        max_tokens = item.get("max_output_tokens", 256)
        category = item.get("category", "general")
        
        dataset.append({
            "id": f"{category}_{i}",
            "category": category,
            "prompt": prompt_text,
            "min_output_tokens": 1,
            "max_output_tokens": max_tokens,
            "ignore_eos": False
        })
        
        # Print progress
        print(f"  [{i+1}/{len(prompts_to_use)}] {category}: {prompt_text[:50]}...")
        
    output_file = "benchmark_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Success! Generated {len(dataset)} prompts in '{output_file}'")
    print("=" * 60)
    
    # Print summary by category
    categories = {}
    for item in dataset:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nPrompts by category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    
    return dataset


if __name__ == "__main__":
    create_dataset()