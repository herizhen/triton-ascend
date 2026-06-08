#!/usr/bin/env python3
"""
Translate Markdown files from source directory to target directory.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

SYSTEM_PROMPT = """You are a professional technical documentation translator.
You translate Chinese technical documentation to English.
Keep all code blocks, links, formatting, and technical terms intact.
Use clear, concise, and accurate technical English."""

TRANSLATION_PROMPT = """Translate the following Chinese Markdown content to English.

Rules:
1. Keep ALL Markdown syntax unchanged (headings #, lists -, code blocks ```, links [], images ![], bold **, etc.)
2. Keep code blocks content unchanged (do NOT translate code inside ```)
3. Keep inline code `like this` unchanged
4. Keep URLs, file paths, and variable names unchanged
5. Use professional technical English terminology
6. Maintain the same tone and style
7. Do NOT add any explanation before or after the translation

Here is the content to translate:

{content}"""


class MarkdownTranslator:

    def __init__(self, api_key: str, max_concurrent: int = 2):
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=60.0)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def translate_content(self, content: str, retry: int = 3) -> str:
        if not content or not content.strip():
            return content
        prompt = TRANSLATION_PROMPT.format(content=content)
        for attempt in range(retry):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=4000,
                        temperature=0.3,
                    )
                    result = response.choices[0].message.content
                    return result if result else content
            except Exception as e:
                print(f"    Attempt {attempt+1}/{retry} failed: {e}")
                if attempt == retry - 1:
                    return content
                await asyncio.sleep(2**attempt)
        return content

    async def translate_file(self, src: Path, dst: Path) -> bool:
        try:
            original = src.read_text(encoding="utf-8")
            if not original.strip():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text("", encoding="utf-8")
                return True
            print(f"  Translating: {src.relative_to(Path.cwd())}")
            translated = await self.translate_content(original)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(translated, encoding="utf-8")
            return True
        except Exception as e:
            print(f"  Error on {src}: {e}")
            return False


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        return 1

    src_root = Path(args.source_dir)
    dst_root = Path(args.target_dir)
    if not src_root.exists():
        print(f"Error: source dir {src_root} not found")
        return 1

    md_files = list(src_root.rglob("*.md"))
    if not md_files:
        print("No .md files found")
        return 0

    print(f"Found {len(md_files)} markdown files")
    translator = MarkdownTranslator(api_key, args.max_concurrent)
    tasks = []
    for src in md_files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        tasks.append(translator.translate_file(src, dst))

    results = await asyncio.gather(*tasks)
    success = sum(results)
    print(f"Translation completed: {success}/{len(md_files)} succeeded")
    return 0 if success == len(md_files) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
