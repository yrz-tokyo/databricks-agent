# AI Development Guidelines

このファイルは AI による開発時の標準ガイドラインです。すべてのコード生成・修正はこのガイドラインに従ってください。

**プロジェクトの詳細**: [requirements.md](requirements.md) を参照

---

## 必須スキル

### すべてのタスクで必須

**`ai-dev-kit/databricks-python-sdk`**
- すべての Databricks 操作はこのスキルに従う
- WorkspaceClient の使用
- 環境変数による設定管理
- serving_endpoints.query() による LLM 呼び出し
- 非同期アプリケーションでは `asyncio.to_thread()` でラップ

### タスク別の追加スキル

| タスク | 追加スキル | 理由 |
|--------|-----------|------|
| Agent 開発 | `databricks-python-sdk` のみ | ChatDatabricks, VectorSearchRetrieverTool 使用 |
| FastAPI 開発 | `web-backend` | REST API 設計パターン |
| React 開発 | `web-frontend` | UI コンポーネント設計 |
| Databricks Apps | `databricks-apps` | デプロイメント手順 |

---

## 開発ガイドライン

### 1. Databricks SDK パターン

#### 認証と初期化
```python
from databricks.sdk import WorkspaceClient

# 環境変数または ~/.databrickscfg から自動検出
w = WorkspaceClient()

# 明示的なプロファイル指定
w = WorkspaceClient(profile="MY_PROFILE")
```

#### LLM エンドポイント呼び出し
```python
# ✅ 正しい: serving_endpoints.query() を使用
response = w.serving_endpoints.query(
    name="databricks-claude-3-7-sonnet",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.1,
    max_tokens=500
)
result = response.choices[0].message.content

# ❌ 間違い: 直接 REST API を呼び出す
# requests.post() は使用しない
```

#### 環境変数による設定管理
```python
import os

# ✅ 正しい: 環境変数から取得
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT_NAME")
VS_NAME = os.environ.get("VS_NAME")

# 必須項目の検証
if not LLM_ENDPOINT:
    raise ValueError("LLM_ENDPOINT_NAME environment variable is required")

# ❌ 間違い: ハードコードされた値
# LLM_ENDPOINT = "databricks-claude-3-7-sonnet"
```

#### 非同期アプリケーション (FastAPI等)
```python
import asyncio
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# ✅ 正しい: asyncio.to_thread() でラップ
@app.get("/clusters")
async def list_clusters():
    clusters = await asyncio.to_thread(lambda: list(w.clusters.list()))
    return [{"id": c.cluster_id, "name": c.cluster_name} for c in clusters]

@app.post("/query")
async def run_query(sql: str, warehouse_id: str):
    response = await asyncio.to_thread(
        w.statement_execution.execute_statement,
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="30s"
    )
    return response.result.data_array

# ❌ 間違い: 同期呼び出し（イベントループをブロック）
# @app.get("/clusters")
# async def list_clusters_bad():
#     return list(w.clusters.list())  # イベントループをブロック！
```

**重要**: Databricks SDK は完全に同期的です。非同期アプリケーションでは必ず `asyncio.to_thread()` でラップしてください。

---

### 2. Agent 開発パターン

#### ChatDatabricks による LLM 呼び出し
```python
from databricks_langchain import ChatDatabricks

# 環境変数から endpoint 名を取得
llm = ChatDatabricks(endpoint=os.environ.get("LLM_ENDPOINT_NAME"))

# temperature 指定
insight_llm = ChatDatabricks(
    endpoint=os.environ.get("LLM_ENDPOINT_NAME"),
    temperature=0.3
)
```

#### VectorSearchRetrieverTool の使用
```python
from databricks_langchain import VectorSearchRetrieverTool

vs_tool = VectorSearchRetrieverTool(
    index_name=os.environ.get("VS_NAME"),
    tool_name="search_product_docs",
    num_results=5,
    tool_description="このツールを使用してIR資料から関連情報を検索します。",
    disable_notice=True
)
```

#### MLflow ChatAgent パターン
```python
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

class LangGraphChatAgent(ChatAgent):
    """MLflow ChatAgent として使用するラッパー"""

    def __init__(self, agent):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # LangGraph の実行ロジック
        request = {"messages": self._convert_messages_to_dict(messages)}
        response_messages = []

        for event in self.agent.stream(request, stream_mode="updates"):
            # メッセージ処理
            pass

        return ChatAgentResponse(messages=response_messages)

# MLflow に登録
mlflow.models.set_model(LangGraphChatAgent(agent_graph))
```

---

### 3. コーディング規約

#### ファイルヘッダー
すべての Python ファイルには以下を含める:
```python
"""
[ファイル名] - [目的の簡潔な説明]

このファイルは [フレームワーク/パターン] に準拠した [機能] の実装です。

開発ガイドライン (ai-dev-kit/databricks-python-sdk skill):
- [使用するパターン1]
- [使用するパターン2]
- [使用するパターン3]
"""
```

#### 設定管理
```python
def get_env_config():
    """環境変数から必要な設定を取得"""

    # 必須の環境変数を取得
    llm_endpoint = os.environ.get("LLM_ENDPOINT_NAME")
    vs_name = os.environ.get("VS_NAME")

    # 必須項目の検証
    if not llm_endpoint:
        raise ValueError("LLM_ENDPOINT_NAME environment variable is required")

    if not vs_name:
        raise ValueError("VS_NAME environment variable is required")

    return {
        "llm_endpoint": llm_endpoint,
        "vs_name": vs_name
    }

# 設定を取得
config = get_env_config()
```

#### エラーハンドリング
```python
from databricks.sdk.errors import NotFound, PermissionDenied

try:
    cluster = w.clusters.get(cluster_id="...")
except NotFound:
    print("Cluster not found")
except PermissionDenied:
    print("Access denied")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

### 4. MLflow と評価

#### 4.1 MLflow 3 GenAI Evaluate API（推奨）

**重要**: Agent 評価には必ず `mlflow.genai.evaluate()` API を使用してください。これにより MLflow UI の「審査」「データセット」タブに評価結果が表示されます。

##### スコアラー関数の定義
```python
import mlflow
from typing import Dict, Any

@mlflow.scorer(name="accuracy_scorer")
def accuracy_scorer(
    predictions: str,
    inputs: Dict[str, Any]
) -> int:
    """
    正確性スコアラー (1-5)

    Args:
        predictions: Agent の出力（文字列）
        inputs: 入力データ（question, expected_topics 等）

    Returns:
        int: スコア (1-5)
    """
    question = inputs.get("question", "")

    system_prompt = """評価基準を定義..."""
    user_prompt = f"質問: {question}\n\n回答: {predictions}\n\n評価してください。"

    try:
        # ai-dev-kit skill パターン: w.serving_endpoints.query() を使用
        response = w.serving_endpoints.query(
            name=LLM_ENDPOINT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        content = response.choices[0].message.content
        # JSON パース前にクリーニング
        cleaned = clean_json_response(content)
        result = json.loads(cleaned)
        return result.get("score", 0)
    except Exception as e:
        print(f"Scorer Error: {e}")
        return 0
```

##### Agent の予測関数
```python
def agent_predict(inputs: Dict[str, Any]) -> str:
    """Agent の予測を実行して文字列で返す"""
    model_input = {
        "messages": [
            {"role": "user", "content": inputs["question"]}
        ]
    }
    result = AGENT.predict(model_input)

    # レスポンスを文字列に変換
    full_response = ""
    if "messages" in result:
        for msg in result["messages"]:
            if msg.get("role") == "assistant":
                full_response += msg.get("content", "") + "\n\n"

    return full_response.strip()
```

##### 評価の実行
```python
import pandas as pd
import mlflow

# 評価データを DataFrame として準備
eval_data = pd.DataFrame([
    {
        "question": "2025年度の売上高は？",
        "expected_topics": ["売上高", "前年比", "単位"]
    },
    {
        "question": "営業利益率の推移は？",
        "expected_topics": ["営業利益率", "推移", "改善"]
    }
])

# mlflow.genai.evaluate() を実行
with mlflow.start_run(run_name="agent_evaluation") as run:
    results = mlflow.genai.evaluate(
        model=agent_predict,  # Agent の予測関数
        data=eval_data,  # pandas DataFrame
        model_type="question-answering",
        evaluators=[
            accuracy_scorer,
            completeness_scorer,
            insight_quality_scorer,
            citation_quality_scorer
        ],
        evaluator_config={
            "col_mapping": {
                "inputs": "question"  # DataFrame の列名をマッピング
            }
        }
    )

    # パラメータをログ
    mlflow.log_params({
        "vector_index": VS_INDEX_NAME,
        "llm_endpoint": LLM_ENDPOINT,
        "num_test_cases": len(eval_data)
    })

print(f"✓ Evaluation completed. Run ID: {run.info.run_id}")

# 結果を表示
metrics_df = results.metrics
print(f"Accuracy Mean: {metrics_df['accuracy_scorer/mean']:.2f}/5")
print(f"Completeness Mean: {metrics_df['completeness_scorer/mean']:.2f}/5")

# 評価テーブルを表示
eval_table = results.tables["eval_results_table"]
display(eval_table)
```

##### JSON レスポンスのクリーニング
LLM Judge が ```json ブロックで JSON を返すことがあるため、クリーニング関数を使用：
```python
import re

def clean_json_response(content: str) -> str:
    """LLM レスポンスから JSON を抽出"""
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*$', '', content)
    return content.strip()
```

##### MLflow UI での確認
- **審査タブ**: 各スコアラーの定義と実行結果
- **データセットタブ**: 評価に使用したデータセット
- **メトリクスタブ**: scorer_name/mean, scorer_name/variance 等

---

#### 4.2 LLM Judge パターン（レガシー）

**非推奨**: 以下は従来の手動評価パターンです。新規開発では 4.1 の `mlflow.genai.evaluate()` を使用してください。

```python
def judge_function(question: str, response: str, llm_endpoint: str) -> dict:
    """
    LLM Judge 関数

    ai-dev-kit skill パターン: w.serving_endpoints.query() を使用
    """
    system_prompt = """評価基準..."""
    user_prompt = f"質問: {question}\n\n回答: {response}\n\n評価してください。"

    try:
        response = w.serving_endpoints.query(
            name=llm_endpoint,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"score": 0, "reasoning": f"評価エラー: {str(e)}"}
```

##### 手動ロギング（レガシー）
```python
import mlflow

# 評価実行
with mlflow.start_run(run_name="evaluation"):
    # Agent 実行
    result = AGENT.predict(model_input)

    # Judge 評価
    accuracy = judge_accuracy(question, response, LLM_ENDPOINT)

    # メトリクスをログ（UI に表示されない）
    mlflow.log_metrics({
        "accuracy": accuracy["score"],
        "completeness": completeness["score"]
    })
```

---

### 5. ファイル構成

```
intelligent-finance-agent/
├── agent.py                    # Agent 実装（LangGraph + MLflow ChatAgent）
├── 02_Driver.py               # 評価とデプロイ（Databricks Notebook）
├── requirements.txt           # Python 依存関係
└── tests/                     # テストケース

workspace-files/
└── intelligent-finance-agent/
    └── agent.py               # Workspace にアップロードする Agent

frontend/
├── src/
│   ├── components/           # React コンポーネント
│   ├── services/            # API クライアント
│   └── App.tsx              # メインアプリケーション
└── package.json

backend/
├── app/
│   ├── main.py              # FastAPI メインアプリ
│   ├── routers/             # API ルーター
│   └── services/            # ビジネスロジック
└── requirements.txt
```

---

### 6. よくあるパターン

#### Notebook での環境変数設定
```python
# Databricks Notebook (02_Driver.py など)
import os

# 環境変数を設定（agent.py が必要とする）
os.environ["LLM_ENDPOINT_NAME"] = LLM_ENDPOINT
os.environ["VS_NAME"] = VS_INDEX_NAME
os.environ["CATALOG_NAME"] = CATALOG_NAME
os.environ["SCHEMA_NAME"] = SCHEMA_NAME

# agent.py から AGENT をインポート
from agent import AGENT
```

#### Unity Catalog への登録
```python
import mlflow

# モデルをログ
with mlflow.start_run(run_name="finance_agent_model") as run:
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=notebook_dir,  # agent.py のパス
        artifact_path="agent",
        input_example=input_example
    )
    model_uri = f"runs:/{run.info.run_id}/agent"

# Unity Catalog に登録
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=f"{CATALOG_NAME}.{SCHEMA_NAME}.finance_agent"
)
```

---

## チェックリスト

### コード生成・修正時
- [ ] `ai-dev-kit/databricks-python-sdk` skill に準拠しているか
- [ ] `WorkspaceClient` を使用しているか
- [ ] 環境変数から設定を取得しているか
- [ ] LLM 呼び出しは `serving_endpoints.query()` を使用しているか
- [ ] 非同期アプリでは `asyncio.to_thread()` でラップしているか
- [ ] ファイルヘッダーに開発ガイドラインを記載しているか
- [ ] エラーハンドリングを適切に実装しているか

### テスト・評価時
- [ ] `mlflow.genai.evaluate()` API を使用しているか（必須）
- [ ] スコアラー関数に `@mlflow.scorer` デコレーターを付けているか
- [ ] 評価データを pandas DataFrame として準備しているか
- [ ] LLM Judge は `serving_endpoints.query()` を使用しているか
- [ ] JSON レスポンスをクリーニングしているか（```json ブロック対策）
- [ ] MLflow UI の「審査」「データセット」タブで確認できるか

### デプロイ時
- [ ] 環境変数が正しく設定されているか
- [ ] Unity Catalog に登録されているか
- [ ] Model Serving エンドポイントが正常に動作しているか

---

## 参考リンク

- **Databricks Python SDK**: https://databricks-sdk-py.readthedocs.io/en/latest/
- **ai-dev-kit skill**: [../ai-dev-kit/.claude/skills/databricks-python-sdk/skill.md](../ai-dev-kit/.claude/skills/databricks-python-sdk/skill.md)
- **要件定義**: [requirements.md](requirements.md)
- **進捗管理**: [progress_2026-03-10.md](progress_2026-03-10.md)

---

## 更新履歴

- 2026-03-10: 初版作成
  - ai-dev-kit/databricks-python-sdk skill を必須化
  - WorkspaceClient、serving_endpoints.query() パターンを明記
  - 非同期アプリケーション対応を追加
  - MLflow 3 GenAI Evaluate API のドキュメントを追加
    - @mlflow.scorer デコレーターパターン
    - mlflow.genai.evaluate() の使用方法
    - JSON クリーニングパターン
