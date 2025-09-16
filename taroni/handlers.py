# taroni/handlers.py

import sys
import asyncio
from pathlib import Path
from google import genai
from google.genai import types
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import print_formatted_text, prompt
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
import json
import re

# utilsからヘルパー関数をインポート
from .utils import (
    parse_and_read_files_from_prompt, add_citations,
    load_chat_history, save_chat_history, PathCompleter
)
# 新しく作成したツールをインポート
from .tools import propose_file_modifications, ModificationPlan, FileModification

# --- 思考表示ヘルパー関数 (変更なし) ---
def display_thought_process(chunk, is_first_thought_ref=[True]):
    if not chunk.candidates or not hasattr(chunk.candidates[0].content, 'parts'): return
    for part in chunk.candidates[0].content.parts:
        if hasattr(part, 'thought') and part.thought and hasattr(part, 'text') and part.text:
            if is_first_thought_ref[0]: print(f"\n--- AIの思考プロセス ---", file=sys.stderr); is_first_thought_ref[0] = False
            print(part.text, end="", flush=True, file=sys.stderr)
def get_final_answer_from_chunk(chunk):
    answer_text = ""
    if chunk.candidates and hasattr(chunk.candidates[0].content, 'parts'):
         for part in chunk.candidates[0].content.parts:
             if (not hasattr(part, 'thought') or not part.thought) and hasattr(part, 'text') and part.text:
                 answer_text += part.text
    return answer_text
def reconstruct_response_from_chunks(chunks: list[types.GenerateContentResponse]) -> types.GenerateContentResponse | None:
    if not chunks: return None
    final_text = "".join(get_final_answer_from_chunk(c) for c in chunks)
    final_metadata = None
    for chunk in reversed(chunks):
        if chunk.candidates and hasattr(chunk.candidates[0], 'grounding_metadata') and chunk.candidates[0].grounding_metadata:
            final_metadata = chunk.candidates[0].grounding_metadata
            break
    final_candidate = types.Candidate(content=types.Content(parts=[types.Part(text=final_text)], role="model"), grounding_metadata=final_metadata)
    return types.GenerateContentResponse(prompt_feedback=chunks[-1].prompt_feedback if chunks else None, candidates=[final_candidate])
# --- ここまで変更なし ---

# ==============================================================================
#  エージェント機能のコアロジック (変更なし)
# ==============================================================================
async def plan_modifications(
    client: genai.Client, model_name: str, system_instruction: str | None, conversation_history: list[types.Content]
) -> tuple[ModificationPlan | None, types.Content | None]:
    agent_system_prompt = (
        "あなたは優秀なAI開発エージェントです。\n"
        "ユーザーからの開発指示や対話を理解し、それを達成するための具体的なファイル変更計画を立案してください。\n"
        "計画は `propose_file_modifications` 関数を呼び出して提案する必要があります。\n"
        "ファイルは必ず丸ごと書き換える前提で計画を立ててください。\n"
    )
    full_system_instruction = f"{agent_system_prompt}\n---\n\n{system_instruction}" if system_instruction else agent_system_prompt
    config = types.GenerateContentConfig(
        tools=[propose_file_modifications],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode='ANY')),
        system_instruction=full_system_instruction
    )
    print("\n[AGENT_STATUS] AIに計画の立案/修正をリクエスト中...", file=sys.stderr)
    response = await client.aio.models.generate_content(model=model_name, contents=conversation_history, config=config)
    if not response.function_calls:
        print("[AGENT_ERROR] AIが計画を提案できませんでした（Function Callがありません）。", file=sys.stderr)
        print("AIの応答:", response.text, file=sys.stderr)
        return None, None
    function_call = response.function_calls[0]
    if function_call.name != "propose_file_modifications":
        print(f"[AGENT_ERROR] AIが予期しない関数({function_call.name})を呼び出しました。", file=sys.stderr)
        return None, None
    try:
        plan_data = function_call.args['plan']
        modification_plan = ModificationPlan.model_validate(plan_data)
        return modification_plan, response.candidates[0].content
    except Exception as e:
        print(f"[AGENT_ERROR] AIが提案した計画の解析に失敗しました: {e}", file=sys.stderr)
        print("--- AIが生成した生の計画データ ---", file=sys.stderr)
        print(json.dumps(function_call.args, indent=2, ensure_ascii=False), file=sys.stderr)
        print("---------------------------------", file=sys.stderr)
        return None, None
def display_plan(plan: ModificationPlan):
    print("\n✅ AIが以下の開発計画を提案しました。")
    print("="*40)
    print(f"思考プロセス: {plan.thoughts}")
    print("-"*40)
    for i, mod in enumerate(plan.modifications, 1):
        print(f"【ステップ {i}】")
        print(f"  ファイル: {mod.file_path}")
        print(f"  アクション: {mod.action}")
        print(f"  変更理由: {mod.reason}")
        print(f"  変更概要: {mod.summary_of_changes}")
    print("="*40)
async def get_user_approval(plan: ModificationPlan) -> tuple[str, str] | None:
    display_plan(plan)
    user_input = await asyncio.to_thread(prompt, "この計画を実行しますか？ (y/n/修正点を指示): ")
    user_input_lower = user_input.strip().lower()
    if user_input_lower in ['y', 'yes', 'はい']: return "APPROVE", user_input
    if user_input_lower in ['n', 'no', 'いいえ']: return "REJECT", user_input
    return "REPLAN", user_input
async def generate_file_content(
    client: genai.Client, model_name: str, system_instruction: str | None,
    modification: FileModification, conversation_history: list[types.Content]
) -> str:
    code_gen_system_prompt = (
        "あなたはプロのソフトウェアエンジニアです。\n"
        "これから渡される指示と会話履歴に基づき、指定されたファイルの新しい内容を**全文**生成してください。\n"
        "あなたの応答には、ファイルの内容**以外**の余計な解説（「はい、承知しました。」「以下がコードです。」など）や、Markdownのコードブロック装飾（```python ... ```）を**絶対に含めないでください**。\n"
        "応答の最初から最後までが、そのままファイルに書き込まれる内容でなければなりません。"
    )
    full_system_instruction = f"{code_gen_system_prompt}\n---\n\n{system_instruction}" if system_instruction else code_gen_system_prompt
    current_content = ""
    file_path = Path(modification.file_path)
    if modification.action == "OVERWRITE" and file_path.exists() and file_path.is_file():
        try: current_content = file_path.read_text(encoding='utf-8')
        except Exception as e: print(f"[AGENT_WARNING] ファイル '{modification.file_path}' の読み込みに失敗しました: {e}", file=sys.stderr)
    generation_prompt = (
        f"ファイルパス '{modification.file_path}' の新しい内容を生成してください。\n"
        f"操作種別: {modification.action}\n"
        f"変更理由: {modification.reason}\n"
        f"変更概要: {modification.summary_of_changes}\n\n"
        "これが現在のファイル内容です（空の場合は新規作成を意味します）:\n"
        "--- CURRENT FILE CONTENT ---\n"
        f"{current_content}\n"
        "--- END OF CURRENT FILE CONTENT ---\n\n"
        "以上の情報とこれまでの会話履歴をすべて考慮して、ファイルの**新しい全文**を生成してください。"
    )
    temp_conversation = conversation_history + [types.Content(role="user", parts=[types.Part(text=generation_prompt)])]
    config = types.GenerateContentConfig(system_instruction=full_system_instruction, temperature=0.1)
    response = await client.aio.models.generate_content(model=model_name, contents=temp_conversation, config=config)
    return response.text
async def execute_plan(
    client: genai.Client, model_name: str, system_instruction: str | None,
    plan: ModificationPlan, conversation_history: list[types.Content]
):
    print("\n[AGENT_STATUS] 承認された計画を実行します...")
    successful_modifications, failed_modifications = [], []
    for i, mod in enumerate(plan.modifications, 1):
        print(f"\n--- ステップ {i}/{len(plan.modifications)}: {mod.action} '{mod.file_path}' ---")
        try:
            print("[AGENT_STATUS] AIに新しいファイル内容を生成させています...", file=sys.stderr)
            new_content = await generate_file_content(client, model_name, system_instruction, mod, conversation_history)
            file_path = Path(mod.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_content, encoding='utf-8')
            print(f"✅ ファイル '{mod.file_path}' の更新に成功しました。")
            successful_modifications.append(mod.file_path)
        except Exception as e:
            print(f"❌ ファイル '{mod.file_path}' の操作中にエラーが発生しました: {e}", file=sys.stderr)
            failed_modifications.append((mod.file_path, str(e)))
    print("\n" + "="*20 + " 実行結果 " + "="*20)
    print("全ての計画ステップが完了しました。")
    if successful_modifications:
        print("\n成功したファイル:"); [print(f"  - {path}") for path in successful_modifications]
    if failed_modifications:
        print("\n失敗したファイル:"); [print(f"  - {path} (エラー: {error})") for path, error in failed_modifications]
    combined_text = " ".join([m.summary_of_changes for m in plan.modifications])
    if "requirements.txt" in successful_modifications:
        print("\n推奨される次のアクション:")
        print("  - `pip install -r requirements.txt` を実行して、新しい依存関係をインストールしてください。")
    if re.search(r'model|database|migrate', combined_text, re.IGNORECASE):
        print("\n推奨される次のアクション:")
        print("  - `python manage.py makemigrations` と `python manage.py migrate` を実行して、データベースの変更を適用してください。")
    print("="*52)
async def run_agent_session(
    client: genai.Client, model_name: str, system_instruction: str | None,
    initial_instruction: str
):
    print("\n--- AI開発エージェントセッション開始 ---")
    processed_instruction = parse_and_read_files_from_prompt(initial_instruction)
    if not processed_instruction.strip():
        print("エラー: 指示が空です。", file=sys.stderr); return
    print("--- ユーザーからの初期指示 ---"); print(processed_instruction); print("---------------------------\n")
    conversation_history = [types.Content(role="user", parts=[types.Part(text=processed_instruction)])]
    try:
        while True:
            plan, ai_response_content = await plan_modifications(client, model_name, system_instruction, conversation_history)
            if not plan:
                print("[AGENT_STATUS] 計画の立案に失敗したため、セッションを中断します。"); return
            conversation_history.append(ai_response_content)
            action, user_feedback = await get_user_approval(plan)
            if action == "APPROVE":
                approved_plan = plan; break
            elif action == "REJECT":
                print("計画が却下されました。セッションを中断します。"); return
            elif action == "REPLAN":
                print("\n🔄 計画の修正をAIに依頼します...")
                conversation_history.append(types.Content(role="user", parts=[types.Part(text=user_feedback)]))
                continue
        await execute_plan(client, model_name, system_instruction, approved_plan, conversation_history)
        print("--- AI開発エージェントセッション終了 ---\n")
    except Exception as e:
        print(f"\n[AGENT_FATAL_ERROR] エージェントの処理中に致命的なエラーが発生しました: {e}", file=sys.stderr)
# --- ここまで変更なし ---

# ==============================================================================
#  モードごとのハンドラ (ここからがモード別の処理)
# ==============================================================================

async def handle_one_shot_mode(client: genai.Client, model_name: str, args, system_instruction: str | None):
    """ワンショット実行モード（エージェントモード）の処理を行う"""
    user_initial_instruction = args.prompt or sys.stdin.read()
    await run_agent_session(client, model_name, system_instruction, user_initial_instruction)


async def handle_interactive_mode(client: genai.Client, model_name: str, args, system_instruction: str | None):
    """対話モードの処理を行う。通常のチャットとエージェント機能を両立させる。"""
    print("--- Interactive Chat Mode ---")
    print("通常の会話ができます。")
    print("'/dev' または '/agent' で開発タスクを開始します。(例: /dev ログイン機能を追加して @app/views.py)")
    print("複数行の入力は [Ctrl+S] で確定。終了は 'exit' または 'quit'。")
    
    chat_history_list = load_chat_history()
    
    try:
        # チャットモード用のAI設定（Google検索ツールを有効化など）
        # エージェントのシステムプロンプトとは独立させる
        chat_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            system_instruction=system_instruction # ユーザーが指定したTARONI.mdなどをチャットの性格付けに使う
        )
        # 新SDKではチャットセッションの履歴は手動で管理する必要がある
        
        session = PromptSession(completer=PathCompleter())
        bindings = KeyBindings()
        @bindings.add('c-s')
        def _(event): event.app.current_buffer.validate_and_handle()

        while True:
            try:
                user_input = await session.prompt_async("You: ", multiline=True, key_bindings=bindings)
                
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("対話を終了します。"); break
                
                # --- エージェントモードへの切り替えを判定 ---
                if user_input.strip().startswith(("/dev", "/agent")):
                    instruction = re.sub(r'^/(dev|agent)\s*', '', user_input, count=1)
                    # エージェントを呼び出す際は、チャットの履歴ではなく、独立したセッションとして実行
                    await run_agent_session(client, model_name, system_instruction, instruction)
                    print("--- Interactive Chat Mode に戻りました ---")
                    continue
                
                # --- ここからが通常のチャット処理 ---
                processed_input = parse_and_read_files_from_prompt(user_input)
                if not processed_input.strip(): continue

                # ユーザーの入力をチャット履歴に追加
                user_content_dict = {"role": "user", "parts": [{"text": processed_input}]}
                chat_history_list.append(user_content_dict)
                
                print_formatted_text(FormattedText([('italic', '...AIに問い合わせ中...')]))

                # 新SDKでは毎回履歴を渡してAPIを呼ぶ
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=chat_history_list,
                    config=chat_config
                )
                
                # AIの応答を表示
                final_text = await add_citations(response) # 引用付与も可能
                print_formatted_text(FormattedText([('bold cyan', 'AI:'), ('', f' {final_text}')]))

                # AIの応答をチャット履歴に追加
                # response.candidates[0].content にはAIの応答全体が含まれている
                if response.candidates:
                    #chat_history_list.append(response.candidates[0].content)
                    #chat_history_list.append(types.Content(role="model", parts=[types.Part(text=final_text)]))
                    user_content_dict = {"role": "model", "parts": [{"text": final_text}]}
                    chat_history_list.append(user_content_dict)
                
            except KeyboardInterrupt:
                print("\n対話を終了します。"); break
    finally:
        print("[INFO] セッション終了時に最終的なチャット履歴を保存します。", file=sys.stderr)
        print(chat_history_list)
        print('chat_history_list')
        save_chat_history(chat_history_list)
