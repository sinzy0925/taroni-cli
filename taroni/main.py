# taroni/main.py

import argparse
import sys
import os
import re
import glob
import asyncio
import json
from google import genai
from google.genai import types
from datetime import datetime
from zoneinfo import ZoneInfo
from .api_key_manager import api_key_manager
import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings

# --- グローバル設定 ---
HISTORY_FILE = "chat_history.json"
MAX_HISTORY_MESSAGES = 100

# --- (ヘルパー関数群は一切変更ありません) ---
async def resolve_redirect(client, url):
    try:
        response = await client.head(url, follow_redirects=True, timeout=5.0)
        return str(response.url)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        print(f"\n[警告] URLの解決に失敗しました: {url}, エラー: {e}", file=sys.stderr)
        return url
async def add_citations(response):
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
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
            if isinstance(history, list):
                print(f"[INFO] {len(history)}件のチャット履歴をロードしました。", file=sys.stderr)
                return history
    except (IOError, json.JSONDecodeError) as e:
        print(f"[警告] チャット履歴ファイルの読み込みに失敗しました: {e}", file=sys.stderr)
    return []
def save_chat_history(history_list: list[dict]):
    try:
        history_to_save = history_list
        if len(history_to_save) > MAX_HISTORY_MESSAGES:
            history_to_save = history_to_save[-MAX_HISTORY_MESSAGES:]
            print(f"[INFO] 履歴が{MAX_HISTORY_MESSAGES}件を超えたため、古いものを削除しました。", file=sys.stderr)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"[エラー] チャット履歴の保存に失敗しました: {e}", file=sys.stderr)
async def handle_one_shot_mode(client: genai.Client, model_name: str, args: argparse.Namespace):
    user_input = args.prompt or sys.stdin.read()
    processed_input = parse_and_read_files_from_prompt(user_input)
    if not processed_input.strip():
        print("エラー: 入力が空です。", file=sys.stderr)
        return
    try:
        config_dict = {'tools': [types.Tool(google_search=types.GoogleSearch())]}
        if args.system_instruction: config_dict['system_instruction'] = args.system_instruction
        generation_kwargs = {'model': model_name, 'contents': [processed_input], 'config': types.GenerateContentConfig(**config_dict)}
        print("...AIからの応答を待っています...", file=sys.stderr)
        if args.stream:
            response_stream = await client.aio.models.generate_content_stream(**generation_kwargs)
            async for chunk in response_stream: print(chunk.text, end="", flush=True)
            print()
        else:
            response = await client.aio.models.generate_content(**generation_kwargs)
            final_text = await add_citations(response)
            print(final_text)
    except Exception as e:
        print(f"\n[One-shot Mode Error] エラーが発生しました: {e}", file=sys.stderr)

# ★★★ ここが最終形態です ★★★
async def handle_interactive_mode(client: genai.Client, model_name: str, args: argparse.Namespace):
    """対話モードの処理を行う (sendkeyアプローチによる最終版)"""
    print("--- Interactive Chat Mode (履歴機能有効) ---")
    print("複数行の入力が可能です。入力を確定するには [Ctrl+Enter] を押してください。")
    print("終了するには 'exit' または 'quit' と入力して確定してください。")
    
    history_list = load_chat_history()
    try:
        config_dict = {'tools': [types.Tool(google_search=types.GoogleSearch())]}
        if args.system_instruction: config_dict['system_instruction'] = args.system_instruction
        chat = client.chats.create(model=model_name, history=history_list, config=types.GenerateContentConfig(**config_dict) if config_dict else None)
        session = PromptSession(completer=PathCompleter())
        
        bindings = KeyBindings()

        # Ctrl+Enter (c-return) が押されたときの処理
        @bindings.add('c-i', eager=True)
        def _(event):
            """
            現在の入力を確定させる。
            これは、バッファの末尾に改行を挿入し、
            その改行を「エンターキー」として処理させることで実現する。
            """
            event.app.current_buffer.insert_text('\n')
            event.app.current_buffer.validate_and_handle()

        while True:
            try:
                # 'multiline=True' と 'c-return' の組み合わせで、
                # 通常のEnterは改行、Ctrl+Enterは確定として機能する
                user_input = await session.prompt_async("You: ", multiline=True, key_bindings=bindings)
                
                # `user_input`には末尾の改行が含まれているので、それを取り除く
                processed_input = parse_and_read_files_from_prompt(user_input.strip())

                if processed_input.strip().lower() in ["exit", "quit"]:
                    print("対話を終了します。")
                    break
                if not processed_input.strip(): continue
                user_content_dict = {
                    "role": "user",
                    "parts": [{"text": processed_input}]
                }
                history_list.append(user_content_dict)

                #history_list.append(types.Content.to_dict(types.Content(parts=[types.Part(text=processed_input)], role="user")))
                print_formatted_text(FormattedText([('italic', '...AIに問い合わせ中...')]))
                response = await asyncio.to_thread(chat.send_message, processed_input)
                #history_list.append(types.Content.to_dict(response.candidates[0].content))
                #final_text = await add_citations(response)
                #print_formatted_text(FormattedText([('bold cyan', 'AI:'), ('', f' {final_text}')]))
                
                ai_content_obj = response.candidates[0].content
                ai_content_dict = {
                    "role": ai_content_obj.role,
                    "parts": [{"text": part.text} for part in ai_content_obj.parts]
                }
                history_list.append(ai_content_dict)
                
                final_text = await add_citations(response)
                print_formatted_text(FormattedText([('bold cyan', 'AI:'), ('', f' {final_text}')]))


            except KeyboardInterrupt:
                print("\n対話を終了します。")
                break
    finally:
        print("[INFO] セッション終了時に最終的なチャット履歴を保存します。", file=sys.stderr)
        save_chat_history(history_list)

# --- (メインロジックは一切変更ありません) ---
async def main():
    parser = argparse.ArgumentParser(description="Gemini APIをコマンドラインから利用するツール 'taroni'", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("prompt", nargs="?", default=None, help="AIへの指示（プロンプト）。指定しない場合は対話モードを開始します。")
    parser.add_argument("--m", "--model", dest="model", default="models/gemini-2.5-pro", help="使用するAIモデルを指定します。")
    parser.add_argument("-S", "--system", dest="system_instruction", default=None, help="システムプロンプトを設定します。")
    parser.add_argument("-s", "--stream", action="store_true", help="結果をストリーミングでリアルタイムに出力します。(この場合、引用は付与されません)")
    args = parser.parse_args()
    api_key = await api_key_manager.get_next_key()
    if not api_key: print("エラー: .env ファイルから有効なAPIキーを読み込めませんでした。", file=sys.stderr); sys.exit(1)
    api_key_manager.save_session()
    jst = ZoneInfo("Asia/Tokyo")
    now_jst = datetime.now(jst)
    timestamp = now_jst.strftime('%y%m%d-%H%M%S')
    print(f"--- Timestamp (JST): {timestamp} ---", file=sys.stderr)
    model_name = args.model
    if not model_name.startswith("models/"): model_name = f"models/{model_name}"
    paid_only_models = ["models/gemini-2.5-flash-image-preview"]
    if model_name in paid_only_models:
        print(f"情報: モデル '{model_name.replace('models/', '')}' は、API経由での利用に課金設定が必要です。", file=sys.stderr)
        sys.exit(0)
    client = genai.Client(api_key=api_key)
    print(f"--- Using Model: {model_name.replace('models/', '')} ---", file=sys.stderr)
    is_one_shot = args.prompt or not sys.stdin.isatty()
    try:
        if is_one_shot:
            await handle_one_shot_mode(client, model_name, args)
        else:
            await handle_interactive_mode(client, model_name, args)
    except Exception as e:
        print(f"\n[FATAL] 予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)
def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n対話を終了します。")
        sys.exit(1)
if __name__ == '__main__':
    run()