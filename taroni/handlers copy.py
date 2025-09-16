# taroni/handlers.py

import sys
import asyncio
from pathlib import Path
from google import genai
from google.genai import types
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings

# utilsからヘルパー関数をインポート
from .utils import (
    parse_and_read_files_from_prompt, add_citations,
    load_chat_history, save_chat_history, PathCompleter
)

# --- 思考表示ヘルパー関数 ---
def display_thought_process(chunk, is_first_thought_ref=[True]):
    """ストリームのチャンクから思考プロセスを抽出し、標準エラー出力に表示する"""
    if not chunk.candidates or not hasattr(chunk.candidates[0].content, 'parts'):
        return
    
    for part in chunk.candidates[0].content.parts:
        if hasattr(part, 'thought') and part.thought and hasattr(part, 'text') and part.text:
            if is_first_thought_ref[0]:
                print(f"\n--- AIの思考プロセス ---", file=sys.stderr)
                is_first_thought_ref[0] = False
            print(part.text, end="", flush=True, file=sys.stderr)

def get_final_answer_from_chunk(chunk):
    """ストリームのチャンクから最終的な回答部分だけを抽出する"""
    answer_text = ""
    if chunk.candidates and hasattr(chunk.candidates[0].content, 'parts'):
         for part in chunk.candidates[0].content.parts:
             if (not hasattr(part, 'thought') or not part.thought) and hasattr(part, 'text') and part.text:
                 answer_text += part.text
    return answer_text

# ★★★ ここからが最後の修正点 ★★★
def reconstruct_response_from_chunks(chunks: list[types.GenerateContentResponse]) -> types.GenerateContentResponse | None:
    """チャンクのリストから単一のGenerateContentResponseを再構築する"""
    if not chunks:
        return None

    # 最終的なテキストとメタデータを収集
    final_text = "".join(get_final_answer_from_chunk(c) for c in chunks)
    final_metadata = None
    
    # 最後のチャンクがメタデータを持っていることが多い
    for chunk in reversed(chunks):
        if chunk.candidates and hasattr(chunk.candidates[0], 'grounding_metadata') and chunk.candidates[0].grounding_metadata:
            final_metadata = chunk.candidates[0].grounding_metadata
            break

    # 最終的なレスポンスオブジェクトを模倣して作成
    # 必要な属性だけを持つ簡易的なオブジェクトで十分
    final_candidate = types.Candidate(
        content=types.Content(
            parts=[types.Part(text=final_text)],
            role="model"
        ),
        grounding_metadata=final_metadata
    )
    
    return types.GenerateContentResponse(
        prompt_feedback=chunks[-1].prompt_feedback if chunks else None,
        candidates=[final_candidate]
    )
# ★★★ ここまでが最後の修正点 ★★★

async def handle_one_shot_mode(client: genai.Client, model_name: str, args, system_instruction: str | None):
    """ワンショット実行モードの処理を行う"""
    user_input = args.prompt or sys.stdin.read()
    processed_input = parse_and_read_files_from_prompt(user_input)
    if not processed_input.strip():
        print("エラー: 入力が空です。", file=sys.stderr)
        return
    try:
        config_dict = {
            'tools': [types.Tool(google_search=types.GoogleSearch())],
            'thinking_config': types.ThinkingConfig(thinking_budget=-1, include_thoughts=True)
        }
        if system_instruction:
            config_dict['system_instruction'] = system_instruction
        
        generation_kwargs = {
            'model': model_name,
            'contents': [processed_input],
            'config': types.GenerateContentConfig(**config_dict)
        }
        
        print("...AIからの応答を待っています...", file=sys.stderr)
        
        stream = await client.aio.models.generate_content_stream(**generation_kwargs)
        
        full_response_chunks = []
        is_first_thought = [True]

        async for chunk in stream:
            display_thought_process(chunk, is_first_thought_ref=is_first_thought)
            if not args.stream:
                full_response_chunks.append(chunk)

            if args.stream:
                print(get_final_answer_from_chunk(chunk), end="", flush=True)

        if not is_first_thought[0]:
            print("\n---------------------\n", file=sys.stderr)

        if args.stream:
            print()
        else:
            # ★★★ ここが最後の修正点 ★★★
            final_response = reconstruct_response_from_chunks(full_response_chunks)
            if final_response:
                final_text_with_citations = await add_citations(final_response)
                print(final_text_with_citations)
            else:
                print("[WARN] AIから有効な応答が得られませんでした。", file=sys.stderr)
        # ★★★ ここまでが最後の修正点 ★★★
            
    except Exception as e:
        print(f"\n[One-shot Mode Error] エラーが発生しました: {e}", file=sys.stderr)

# (handle_interactive_modeは変更ありません)
async def handle_interactive_mode(client: genai.Client, model_name: str, args, system_instruction: str | None):
    # ... (この関数は変更なし) ...
    print("--- Interactive Chat Mode (履歴/思考表示機能有効) ---")
    print("複数行の入力が可能です。入力を確定するには [Ctrl+S] を押してください。")
    print("終了するには 'exit' または 'quit' と入力して確定してください。")
    
    history_list = load_chat_history()
    try:
        config_dict = {
            'tools': [types.Tool(google_search=types.GoogleSearch())],
            'thinking_config': types.ThinkingConfig(thinking_budget=-1, include_thoughts=True)
        }
        if system_instruction: config_dict['system_instruction'] = system_instruction
        
        chat = client.chats.create(model=model_name, history=history_list, config=types.GenerateContentConfig(**config_dict) if config_dict else None)
        
        session = PromptSession(completer=PathCompleter())
        bindings = KeyBindings()
        @bindings.add('c-s')
        def _(event): event.app.current_buffer.validate_and_handle()
        while True:
            try:
                user_input = await session.prompt_async("You: ", multiline=True, key_bindings=bindings)
                processed_input = parse_and_read_files_from_prompt(user_input)
                if processed_input.strip().lower() in ["exit", "quit"]: print("対話を終了します。"); break
                if not processed_input.strip(): continue
                
                user_content_dict = {"role": "user", "parts": [{"text": processed_input}]}
                history_list.append(user_content_dict)
                print_formatted_text(FormattedText([('italic', '...AIに問い合わせ中...')]))
                response = await asyncio.to_thread(chat.send_message, processed_input)
                
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought and hasattr(part, 'text') and part.text:
                         print_formatted_text(FormattedText([('gray', f"\n[思考]\n{part.text}\n")]))
                
                final_text = await add_citations(response)
                print_formatted_text(FormattedText([('bold cyan', 'AI:'), ('', f' {final_text}')]))
                
                user_content_dict = {"role": "model", "parts": [{"text": final_text}]}
                history_list.append(user_content_dict)
            except KeyboardInterrupt:
                print("\n対話を終了します。")
                break
    finally:
        print("[INFO] セッション終了時に最終的なチャット履歴を保存します。", file=sys.stderr)
        save_chat_history(history_list)
