# taroni/tools.py

from typing import List, Literal
from pydantic import BaseModel, Field

class FileModification(BaseModel):
    """単一のファイルに対する変更操作を表すモデル"""
    file_path: str = Field(..., description="操作対象となるプロジェクトルートからの相対パス。")
    action: Literal["CREATE", "OVERWRITE"] = Field(..., description="ファイルに対する操作種別。新規作成は'CREATE'、既存ファイルの上書きは'OVERWRITE'。")
    reason: str = Field(..., description="このファイル操作が必要な理由についての簡潔な説明。")
    summary_of_changes: str = Field(..., description="このファイルで変更または追加される主要なクラスや関数の概要。")

class ModificationPlan(BaseModel):
    """ファイル変更計画全体を表すモデル"""
    modifications: List[FileModification] = Field(..., description="実行すべき一連のファイル変更操作のリスト。")
    thoughts: str = Field(..., description="この計画を立案した際のAIの思考プロセスや全体的な方針。")

def propose_file_modifications(plan: ModificationPlan):
    """
    アプリケーションを新規作成または修正するためのファイル変更計画を提案します。
    ユーザーの指示を達成するために必要な全てのファイル操作をこの関数で提案してください。
    """
    # この関数はAIに構造を理解させるためのもので、実際には呼び出されません。
    # そのため、中身は空でpassします。
    pass