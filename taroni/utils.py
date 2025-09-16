# taroni/utils.py

import os
import re
import glob
import json
import asyncio
from pathlib import Path
import httpx
from prompt_toolkit.completion import Completer, Completion
from google import genai
from google.genai import types
import sys

# --- グローバル設定 ---
HISTORY_FILE = "chat_history.json"
MAX_HISTORY_MESSAGES = 100

async def resolve_redirect(client, url):
    """リダイレクトURLをたどって最終的なURLを返す"""
    # (コードはmain.pyからそのまま移動)
    try:
        response = await client.head(url, follow_redirects=True, timeout=5.0)
        return str(response.url)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        print(f"\n[警告] URLの解決に失敗しました: {url}, エラー: {e}", file=sys.stderr)
        return url

async def add_citations(response):
    """API応答の引用URLを解決し、最終的なURLをテキストに埋め込む"""
    # (コードはmain.pyからそのまま移動)
    text = response.text
    if not hasattr(response.candidates[0], 'grounding_metadata') or not response.candidates[0].grounding_metadata: return text
    metadata = response.candidates[0].grounding_metadata
    supports = metadata.grounding_supports
    chunks = metadata.grounding_chunks
    if not supports or not chunks: return text
    original_urls = [chunk.web.uri for chunk in chunks if hasattr(chunk.web, 'uri')]
    if not original_urls: return text
    async with httpx.AsyncClient() as client:
        tasks = [resolve_redirect(client, url) for url in original_urls]
        resolved_urls = await asyncio.gather(*tasks)
    url_map = dict(zip(original_urls, resolved_urls))
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)
    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            citation_links = []
            for i in sorted(support.grounding_chunk_indices):
                if i < len(chunks):
                    if hasattr(chunks[i].web, 'uri'):
                        original_uri = chunks[i].web.uri
                        final_uri = url_map.get(original_uri, original_uri)
                        citation_links.append(f"[{i + 1}]({final_uri})")
            citation_string = "".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]
    return text


def parse_and_read_files_from_prompt(prompt_text):
    """プロンプト内の @filename を見つけ、そのファイルの内容に置き換える。"""
    # (コードはmain.pyからそのまま移動)
    pattern = re.compile(r'@([\w\./\-\\]+)')
    def replacer(match):
        filepath = match.group(1)
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f: content = f.read()
            print(f"[INFO] ファイルを読み込みました: {filepath}", file=sys.stderr)
            return f"\n\n--- ファイル '{filepath}' の内容 ---\n```\n{content}\n```\n--- ファイル '{filepath}' の内容終わり ---\n"
        except FileNotFoundError:
            print(f"[警告] ファイルが見つかりません: {filepath}", file=sys.stderr)
            return match.group(0)
        except Exception as e:
            print(f"[エラー] ファイル読み込み中にエラーが発生しました ({filepath}): {e}", file=sys.stderr)
            return match.group(0)
    return pattern.sub(replacer, prompt_text)

class PathCompleter(Completer):
    """'@'で始まる単語をファイル/ディレクトリパスとして補完するコンプリータ。"""
    # (コードはmain.pyからそのまま移動)
    def get_completions(self, document, complete_event):
        word_before_cursor = document.get_word_before_cursor(WORD=True)
        if word_before_cursor.startswith('@'):
            path_prefix = word_before_cursor[1:]
            try:
                search_path = f"{path_prefix}*"
                for path in glob.glob(search_path):
                    if os.path.isdir(path):
                        display_text = f"{os.path.basename(path)}/"
                        completion_text = f"@{path}/"
                    else:
                        display_text = os.path.basename(path)
                        completion_text = f"@{path}"
                    yield Completion(completion_text, start_position=-len(word_before_cursor), display=display_text, display_meta="file/dir")
            except Exception:
                pass

def load_chat_history() -> list[dict]:
    """チャット履歴ファイルを読み込む"""
    # (コードはmain.pyからそのまま移動)
    history_path = Path.cwd() / HISTORY_FILE
    if not history_path.exists():
        return []
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
            if isinstance(history, list):
                print(f"[INFO] {len(history)}件のチャット履歴をロードしました。", file=sys.stderr)
                return history
    except (IOError, json.JSONDecodeError) as e:
        print(f"[警告] チャット履歴ファイルの読み込みに失敗しました: {e}", file=sys.stderr)
    return []

def save_chat_history(history_list: list[dict]):
    """チャット履歴(辞書のリスト)をファイルに保存する (古いものは削除)"""
    # (コードはmain.pyからそのまま移動)
    history_path = Path.cwd() / HISTORY_FILE
    try:
        history_to_save = history_list
        if len(history_to_save) > MAX_HISTORY_MESSAGES:
            history_to_save = history_to_save[-MAX_HISTORY_MESSAGES:]
            print(f"[INFO] 履歴が{MAX_HISTORY_MESSAGES}件を超えたため、古いものを削除しました。", file=sys.stderr)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"[エラー] チャット履歴の保存に失敗しました: {e}", file=sys.stderr)