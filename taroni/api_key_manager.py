# taroni/api_key_manager.py

import os
import json
import asyncio
from dotenv import load_dotenv
import inspect
from pathlib import Path
import sys

# --- 設定ディレクトリと.envファイルのパスを定義 ---
home_dir = Path.home()
config_dir = home_dir / ".taroni"
config_dir.mkdir(exist_ok=True) # ディレクトリがなければ作成

dotenv_path = config_dir / ".env"

# 指定したパスの.envファイルから環境変数を読み込む
load_dotenv(dotenv_path=dotenv_path)

# セッションファイルはコマンド実行場所（カレントディレクトリ）に作成
SESSION_FILE = Path.cwd() / '.session_data.json'

class ApiKeyManager:
    """
    複数のAPIキーを管理し、安全なローテーション、セッションの永続化、
    および高負荷な並列処理下でのレースコンディションを回避するシステム。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ApiKeyManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        print(f"[{self.__class__.__name__}] 設定ディレクトリ: {config_dir}", file=sys.stderr)
        if dotenv_path.is_file():
            print(f"[{self.__class__.__name__}] .envファイルを読み込みました: {dotenv_path}", file=sys.stderr)
        else:
            print(f"[{self.__class__.__name__}] .envファイルが見つかりません: {dotenv_path}", file=sys.stderr)

        self._api_keys: list[str] = []
        self._current_index: int = -1
        self._key_selection_lock = asyncio.Lock()
        
        self._load_api_keys_from_env()
        self._load_session()
        
        print(f"[{self.__class__.__name__}] 初期化完了。{len(self._api_keys)}個のキーをロードしました。")

    def _load_api_keys_from_env(self):
        keys = set()
        base_key = os.getenv('GOOGLE_API_KEY')
        if base_key: keys.add(base_key)
        
        i = 1
        while True:
            key = os.getenv(f'GOOGLE_API_KEY_{i}')
            if key:
                keys.add(key)
                i += 1
            else:
                break
        
        self._api_keys = list(keys)
        if not self._api_keys:
            print(f"警告: 有効なAPIキーが {dotenv_path} に設定されていません。")

    def _load_session(self):
        try:
            if SESSION_FILE.exists():
                with open(SESSION_FILE, 'r') as f:
                    data = json.load(f)
                    last_index = data.get('lastKeyIndex', -1)
                    if 0 <= last_index < len(self._api_keys):
                        self._current_index = last_index
                        print(f"[{self.__class__.__name__}] セッションをロードしました。次のキーインデックスは { (last_index + 1) % len(self._api_keys) if self._api_keys else 0 } から開始します。")
                    else:
                        self._current_index = -1
        except (IOError, json.JSONDecodeError) as e:
            print(f"セッションファイルの読み込み中にエラーが発生しました: {e}")
            self._current_index = -1

    def save_session(self):
        if not self._api_keys:
            return
        try:
            with open(SESSION_FILE, 'w') as f:
                json.dump({'lastKeyIndex': self._current_index}, f)
        except IOError as e:
            print(f"セッションファイルの保存に失敗しました: {e}")

    async def get_next_key(self) -> str | None:
        if not self._api_keys:
            print("エラー: 利用可能なAPIキーがありません。")
            return None
        try:
            caller_frame = inspect.stack()[1]
            caller_info = f"From: {os.path.basename(caller_frame.filename)}:{caller_frame.lineno}"
        except IndexError:
            caller_info = "呼び出し元: 不明"
        async with self._key_selection_lock:
            if not self._api_keys: return None
            self._current_index = (self._current_index + 1) % len(self._api_keys)
            selected_key = self._api_keys[self._current_index]
            print(f"[{self.__class__.__name__}] APIkey: idx: {self._current_index}, key: ...{selected_key[-4:]} [{caller_info}]")
            return selected_key

    @property
    def last_used_key_info(self) -> dict:
        if self._current_index == -1 or not self._api_keys:
            return {"key_snippet": "N/A", "index": -1, "total": len(self._api_keys)}
        key = self._api_keys[self._current_index]
        return {"key_snippet": key[-4:], "index": self._current_index, "total": len(self._api_keys)}

api_key_manager = ApiKeyManager()