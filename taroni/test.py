# taroni/cli.py

import argparse
import sys
import asyncio
from pathlib import Path
from google import genai
from datetime import datetime
from zoneinfo import ZoneInfo

# 他のモジュールから必要なコンポーネントをインポート
from .api_key_manager import api_key_manager
# ★ handle_one_shot_mode は更新したので、こちらも更新
from .handlers import handle_one_shot_mode, handle_interactive_mode

# 以下、ファイルの他の部分は変更ありません。
# main関数やrun関数のロジックはそのまま利用できます。
async def main():
    """taroniコマンドのメイン非同期処理"""

    # --- 1. 設定ファイルのパスを定義 ---
    home_config_dir = Path.home() / ".taroni"
    home_system_prompt_file = home_config_dir / "TARONI.md"
    current_dir_system_prompt_file = Path.cwd() / "TARONI.md"

    # --- 2. 階層的にデフォルトのシステムプロンプトを決定 ---
    default_system_prompt = None
    if current_dir_system_prompt_file.is_file():
        try:
            default_system_prompt = current_dir_system_prompt_file.read_text(encoding='utf-8')
            print(f"[INFO] プロジェクトのシステムプロンプトを読み込みました: {current_dir_system_prompt_file}", file=sys.stderr)
        except Exception as e:
            print(f"[警告] {current_dir_system_prompt_file} の読み込みに失敗しました: {e}", file=sys.stderr)       
    elif home_system_prompt_file.is_file():
        try:
            default_system_prompt = home_system_prompt_file.read_text(encoding='utf-8')
            print(f"[INFO] デフォルトのシステムプロンプトを読み込みました: {home_system_prompt_file}", file=sys.stderr)
        except Exception as e:
            print(f"[警告] {home_system_prompt_file} の読み込みに失敗しました: {e}", file=sys.stderr)

    # --- 3. argparse の設定 ---
    parser = argparse.ArgumentParser(description="Gemini APIをコマンドラインから利用するツール 'taroni'", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("prompt", nargs="?", default=None, help="AIへの指示（プロンプト）。指定しない場合は対話モードを開始します。")
    parser.add_argument("--m", "--model", dest="model", default="models/gemini-2.5-pro", help="使用するAIモデルを指定します。")
    parser.add_argument("-S", "--system", dest="system_instruction", default=None, help="システムプロンプトを設定します。(全てのTARONI.mdより優先されます)")
    parser.add_argument("-s", "--stream", action="store_true", help="結果をストリーミングでリアルタイムに出力します。(この場合、引用は付与されません)")
    args = parser.parse_args()

    # --- 4. セットアップ処理 ---
    api_key = await api_key_manager.get_next_key()
    if not api_key:
        print("エラー: .env ファイルから有効なAPIキーを読み込めませんでした。", file=sys.stderr)
        sys.exit(1)
    api_key_manager.save_session()

    jst = ZoneInfo("Asia/Tokyo")
    now_jst = datetime.now(jst)
    timestamp = now_jst.strftime('%y%m%d-%H%M%S')
    print(f"--- Timestamp (JST): {timestamp} ---", file=sys.stderr)

    model_name = args.model
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    paid_only_models = ["models/gemini-2.5-flash-image-preview"]
    if model_name in paid_only_models:
        print(f"情報: モデル '{model_name.replace('models/', '')}' は、API経由での利用に課金設定が必要です。", file=sys.stderr)
        sys.exit(0)

    client = genai.Client(api_key=api_key)
    print(f"--- Using Model: {model_name.replace('models/', '')} ---", file=sys.stderr)

    # --- 5. 最終的なシステムプロンプトを決定 ---
    final_system_instruction = args.system_instruction or default_system_prompt

    # --- 6. モード判定と処理の振り分け ---
    is_one_shot = args.prompt or not sys.stdin.isatty()
    try:
        if is_one_shot:
            await handle_one_shot_mode(client, model_name, args, final_system_instruction)
        else:
            await handle_interactive_mode(client, model_name, args, final_system_instruction)
    except Exception as e:
        print(f"\n[FATAL] 予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

def run():
    """コマンドラインからの同期的なエントリーポイント"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        sys.exit(1)

if __name__ == '__main__':
    run()