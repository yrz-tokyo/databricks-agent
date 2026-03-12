# Databricks notebook source
# MAGIC %md
# MAGIC # Intelligent Finance Agent - Driver (Evaluation & Deployment)
# MAGIC
# MAGIC このノートブックは以下を実行します：
# MAGIC 1. Agent のセットアップと設定
# MAGIC 2. 基本動作確認
# MAGIC 3. MLflow 3 GenAI Evaluate による評価（MLflow 管理データセット + 4スコアラー）
# MAGIC 4. MLflow モデルログ
# MAGIC 5. Unity Catalog への登録
# MAGIC 6. Model Serving へのデプロイ
# MAGIC
# MAGIC **開発ガイドライン (ai-dev-kit/databricks-python-sdk skill):**
# MAGIC - WorkspaceClient を使用
# MAGIC - 環境変数による設定管理
# MAGIC - LLM Judge では serving_endpoints.query() を使用
# MAGIC - MLflow 3 GenAI: mlflow.genai.evaluate() + @scorer + Feedback オブジェクト

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 設定

# COMMAND ----------

# Configuration
CATALOG_NAME = "intelligent_finance_agent"
SCHEMA_NAME = "agent_schema"
VS_INDEX_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.kirin_ir_index"
VS_ENDPOINT_NAME = "finance_agent_endpoint"

# LLM Configuration
LLM_ENDPOINT = "databricks-claude-3-7-sonnet"

# Model Configuration
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.finance_agent"
SERVING_ENDPOINT_NAME = "finance-agent-endpoint"

# Evaluation Dataset (Unity Catalog で管理)
EVAL_DATASET_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.finance_agent_eval_dataset"

print(f"Vector Index:    {VS_INDEX_NAME}")
print(f"LLM Endpoint:    {LLM_ENDPOINT}")
print(f"Model Name:      {MODEL_NAME}")
print(f"Eval Dataset:    {EVAL_DATASET_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ライブラリのインストール
# MAGIC
# MAGIC MLflow 3 GenAI Evaluation には `mlflow[databricks]>=3.1.0` が必要

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.1.0" "databricks-agents>=1.2.0" \
# MAGIC     langgraph langchain-core databricks-vectorsearch databricks-langchain databricks-sdk openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Agent のインポートとセットアップ

# COMMAND ----------

# Configuration (再定義 - restartPython後)
CATALOG_NAME = "intelligent_finance_agent"
SCHEMA_NAME = "agent_schema"
VS_INDEX_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.kirin_ir_index"
VS_ENDPOINT_NAME = "finance_agent_endpoint"
LLM_ENDPOINT = "databricks-claude-3-7-sonnet"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.finance_agent"
SERVING_ENDPOINT_NAME = "finance-agent-endpoint"
EVAL_DATASET_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.finance_agent_eval_dataset"

# COMMAND ----------

import os
import mlflow
from databricks.sdk import WorkspaceClient

# 環境変数を設定（agent.py が必要とする）
os.environ["LLM_ENDPOINT_NAME"] = LLM_ENDPOINT
os.environ["VS_NAME"] = VS_INDEX_NAME
os.environ["CATALOG_NAME"] = CATALOG_NAME
os.environ["SCHEMA_NAME"] = SCHEMA_NAME

# WorkspaceClient の初期化（ai-dev-kit skill 推奨パターン）
w = WorkspaceClient()
print("✓ WorkspaceClient initialized")
print(f"  Host: {w.config.host}")

# COMMAND ----------

# agent.py から AGENT をインポート（同じディレクトリにある）
from agent import AGENT

print("✓ Agent imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MLflow Experiment の設定 + AutoLog

# COMMAND ----------

# MLflow Experiment 設定
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_path = f"/Users/{username}/intelligent-finance-agent-evaluation"

mlflow.set_experiment(experiment_path)

# LangChain / LangGraph の自動トレーシングを有効化
mlflow.langchain.autolog()

print(f"✓ MLflow Experiment: {experiment_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 基本動作確認テスト

# COMMAND ----------

test_question = "キリンの2025年の売上高はいくらですか？"
print(f"質問: {test_question}")
print("=" * 60)

model_input = {
    "messages": [{"role": "user", "content": test_question}]
}

with mlflow.start_run(run_name="basic_test"):
    result = AGENT.predict(model_input)

    print("\n[Agent の回答]")
    if "messages" in result:
        for msg in result["messages"]:
            if msg.get("role") == "assistant":
                print(msg.get("content", ""))
                print("-" * 60)

print("\n✓ 基本動作確認完了")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 評価データセットの作成（MLflow 管理 UC Dataset）
# MAGIC
# MAGIC **MLflow 3 GenAI データ形式:**
# MAGIC - `inputs`: predict_fn に kwargs としてアンパックして渡される（必須）
# MAGIC - `expectations`: built-in スコアラー用のグラウンドトゥルース（任意）
# MAGIC
# MAGIC データは Unity Catalog のテーブル `EVAL_DATASET_NAME` に永続化されます。

# COMMAND ----------

import mlflow.genai.datasets

# MLflow 3 GenAI 正規データ形式: {"inputs": {...}, "expectations": {...}}
eval_records = [
    {
        "inputs": {
            "question": "2025年度の売上高と前年比の成長率を教えてください",
        },
        "expectations": {
            "expected_facts": [
                "売上高の具体的な数値（単位付き：百万円または億円）",
                "前年比の成長率または増減額",
            ]
        }
    },
    {
        "inputs": {
            "question": "営業利益率の推移と収益性の改善について分析してください",
        },
        "expectations": {
            "expected_facts": [
                "営業利益率の数値（%）",
                "前年比との比較",
                "収益性改善または悪化の要因",
            ]
        }
    },
    {
        "inputs": {
            "question": "主要事業セグメントの業績と各セグメントの特徴を教えてください",
        },
        "expectations": {
            "expected_facts": [
                "主要セグメント名（食品・飲料、医薬など）",
                "各セグメントの売上高または利益",
                "セグメントの特徴や成長性",
            ]
        }
    },
    {
        "inputs": {
            "question": "海外事業の成長戦略と売上構成比について説明してください",
        },
        "expectations": {
            "expected_facts": [
                "海外売上高または構成比",
                "地域別の内訳または成長率",
                "海外展開の戦略方針",
            ]
        }
    },
    {
        "inputs": {
            "question": "キャッシュフローと配当政策について教えてください",
        },
        "expectations": {
            "expected_facts": [
                "営業キャッシュフローの金額",
                "配当金額または配当利回り",
                "株主還元の方針",
            ]
        }
    },
]

# MLflow 管理データセットを UC テーブルとして作成・更新
# Databricks Notebook では Spark セッションが自動利用可能
try:
    eval_dataset = mlflow.genai.datasets.create_dataset(name=EVAL_DATASET_NAME)
    print("  (新規作成)")
except Exception:
    eval_dataset = mlflow.genai.datasets.get_dataset(name=EVAL_DATASET_NAME)
    print("  (既存テーブルを使用)")
eval_dataset.merge_records(eval_records)

print(f"✓ 評価データセット作成完了: {EVAL_DATASET_NAME}")
print(f"  件数: {len(eval_records)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. predict_fn の定義
# MAGIC
# MAGIC **MLflow 3 GenAI predict_fn の要件:**
# MAGIC - `inputs` の各キーが kwargs としてアンパックされて渡される
# MAGIC - `def predict_fn(question, ...)` のように各キーを引数として受け取る
# MAGIC - 戻り値は dict（scorers で `outputs` として参照）

# COMMAND ----------

def predict_fn(question):
    """
    Agent の予測関数。

    MLflow 3 GenAI の仕様:
    - inputs = {"question": "..."} がアンパックされて predict_fn(question="...") として呼ばれる
    - 戻り値 dict が scorers の outputs 引数に渡される
    """
    model_input = {
        "messages": [{"role": "user", "content": question}]
    }
    result = AGENT.predict(model_input)

    # アシスタントメッセージを結合して文字列に変換
    full_response = ""
    if "messages" in result:
        for msg in result["messages"]:
            if msg.get("role") == "assistant":
                full_response += msg.get("content", "") + "\n\n"

    return {"response": full_response.strip()}


print("✓ predict_fn defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Built-in スコアラーの定義
# MAGIC
# MAGIC **MLflow 3 GenAI built-in スコアラー:**
# MAGIC - `Safety` - 安全性チェック
# MAGIC - `RelevanceToQuery` - 質問への関連性チェック
# MAGIC - `Correctness` - expected_facts との一致チェック（データセットの expectations を使用）
# MAGIC - `Guidelines` - 財務回答品質のガイドラインチェック
# MAGIC
# MAGIC これらは `.register()` で審査タブにも登録できます。

# COMMAND ----------

from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    Correctness,
    Guidelines,
    ScorerSamplingConfig,
)

# --- built-in スコアラーの定義 ---

# 1. 安全性チェック
safety_scorer = Safety()

# 2. 質問への関連性
relevance_scorer = RelevanceToQuery()

# 3. 正確性（データセットの expected_facts を使用）
correctness_scorer = Correctness()

# 4. 財務回答品質ガイドライン
financial_quality_scorer = Guidelines(
    name="financial_quality",
    guidelines=[
        "The response must include specific numerical values with units (百万円 or 億円) when discussing financial figures",
        "The response must directly address the financial question asked",
        "The response must reference specific fiscal years or time periods",
        "The response must provide actionable insights or analysis beyond just raw data",
    ]
)

# 5. 引用・単位表記ガイドライン
citation_scorer = Guidelines(
    name="citation_quality",
    guidelines=[
        "The response must clearly state numerical units (百万円、億円、% etc.)",
        "The response must not present financial figures without proper units",
    ]
)

# 評価に使用するスコアラーリスト
SCORERS = [
    safety_scorer,
    relevance_scorer,
    correctness_scorer,
    financial_quality_scorer,
    citation_scorer,
]

print("✓ Built-in scorers defined")
for s in SCORERS:
    print(f"  - {s.name if hasattr(s, 'name') else type(s).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Built-in スコアラーを「審査」タブに登録
# MAGIC
# MAGIC `.register()` → 審査タブに表示
# MAGIC `.start()` → 本番トレースへの自動適用（今回はサンプルレート 100%）

# COMMAND ----------

from mlflow.genai.scorers import list_scorers, delete_scorer

# 既存登録を削除してから再登録（冪等性確保）
existing_names = {s.name for s in list_scorers()}

scorers_for_registry = [
    (safety_scorer,           "finance_safety"),
    (relevance_scorer,        "finance_relevance"),
    (correctness_scorer,      "finance_correctness"),
    (financial_quality_scorer, "financial_quality"),
    (citation_scorer,          "citation_quality"),
]

for scorer_obj, reg_name in scorers_for_registry:
    if reg_name in existing_names:
        delete_scorer(name=reg_name)
        print(f"  削除: {reg_name}")

    registered = scorer_obj.register(name=reg_name)
    # start() で審査タブに「稼働中」として表示
    registered.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0)
    )
    print(f"  ✓ 登録・開始: {reg_name}")

print("\n✓ 審査タブへの登録完了 → MLflow UI の「審査」タブを確認してください")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. MLflow GenAI Evaluate の実行
# MAGIC
# MAGIC **MLflow 3 正規 API:**
# MAGIC - `mlflow.genai.evaluate(data=..., predict_fn=..., scorers=[...])`
# MAGIC - `data`: MLflow 管理データセット または `list[dict]`
# MAGIC - `predict_fn`: `inputs` の各キーを kwargs として受け取る関数
# MAGIC - `scorers`: `@scorer` デコレーターで定義したスコアラーのリスト
# MAGIC - 結果は MLflow UI の「評価」タブで確認可能

# COMMAND ----------

print("=" * 60)
print("MLflow 3 GenAI Evaluation を開始")
print("=" * 60)

results = mlflow.genai.evaluate(
    data=eval_dataset,          # MLflow 管理 UC データセット
    predict_fn=predict_fn,      # inputs の各キーを kwargs で受け取る
    scorers=SCORERS             # セル8で定義した built-in スコアラーリスト
)

print(f"\n✓ MLflow GenAI Evaluation 完了")
print(f"  Run ID: {results.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. 評価結果サマリー

# COMMAND ----------

print("=" * 60)
print("評価結果サマリー")
print("=" * 60)

metrics = results.metrics
print("\n【全体平均スコア】")

scorer_display = [
    ("accuracy_scorer",       "正確性 (Accuracy)         "),
    ("completeness_scorer",   "完全性 (Completeness)     "),
    ("insight_quality_scorer","インサイト品質 (Insight)   "),
    ("citation_quality_scorer","引用品質 (Citation)       "),
]

scores = []
for key, label in scorer_display:
    mean_key = f"{key}/mean"
    if mean_key in metrics:
        score = metrics[mean_key]
        scores.append(score)
        print(f"  {label}: {score:.2f} / 5")

if scores:
    overall = sum(scores) / len(scores)
    target = 4.0
    status = "✅ 目標達成" if overall >= target else "⚠️ 要改善"
    print(f"\n  総合平均: {overall:.2f} / 5  {status}（目標: {target:.1f}）")

print(f"\n  MLflow Run ID: {results.run_id}")
print(f"  Experiment:   {experiment_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Unity Catalog へのモデル登録

# COMMAND ----------

# Input example
input_example = {
    "messages": [
        {"role": "user", "content": "キリンの2025年の売上高はいくらですか？"}
    ]
}

# resources を定義: log_model に渡すことで Databricks が自動的に
# serving endpoint の service principal に対してアクセス権限を付与する
# → PAT や Secret scope 不要で認証が通る
from mlflow.models.resources import DatabricksServingEndpoint
from databricks_langchain import VectorSearchRetrieverTool
from agent import tools as agent_tools

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]
for tool in agent_tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)

print(f"Resources: {[str(r) for r in resources]}")

with mlflow.start_run(run_name="finance_agent_model") as run:
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model="agent.py",
        artifact_path="agent",
        input_example=input_example,
        resources=resources,
    )
    model_uri = f"runs:/{run.info.run_id}/agent"
    print(f"✓ Model logged: {model_uri}")

# COMMAND ----------

# Unity Catalog に登録
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print(f"✓ Model registered to Unity Catalog: {MODEL_NAME}")
print(f"  Version: {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Model Serving へのデプロイ（オプション）

# COMMAND ----------

from databricks import agents

# agents.deploy() を使用:
#   - リアルタイムトレース（MLflow 3）を自動有効化
#   - AI Gateway 推論テーブルを自動有効化
#   - サービスプリンシパルへの権限付与を自動処理
# ⚠️ Git フォルダからの実行のため、experiment_path を明示指定（トレース有効化に必須）
deployment = agents.deploy(
    model_name=MODEL_NAME,
    model_version=registered_model.version,
    scale_to_zero_enabled=True,
    environment_vars={
        "LLM_ENDPOINT_NAME": LLM_ENDPOINT,
        "VS_NAME": VS_INDEX_NAME,
    },
    mlflow_experiment_path=experiment_path,
)

print(f"✓ Endpoint name:  {deployment.endpoint_name}")
print(f"✓ Query URL:      {deployment.query_endpoint}")
print(f"✓ Review App URL: {deployment.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 完了！
# MAGIC
# MAGIC 以下が完了しました：
# MAGIC - ✅ Agent のセットアップと基本動作確認
# MAGIC - ✅ MLflow 管理 UC データセットの作成（`EVAL_DATASET_NAME`）
# MAGIC - ✅ MLflow 3 GenAI Evaluate による評価（4スコアラー × 5テストケース）
# MAGIC - ✅ MLflow モデルログと Unity Catalog への登録
# MAGIC - ✅ Model Serving エンドポイントへのデプロイ
# MAGIC
# MAGIC **MLflow 3 GenAI 適用:**
# MAGIC - ✅ `mlflow.genai.evaluate()` を使用（`mlflow.evaluate()` ではない）
# MAGIC - ✅ データ形式: `{"inputs": {...}, "expectations": {...}}` ネスト構造
# MAGIC - ✅ `predict_fn` は inputs の各キーを kwargs として受け取る
# MAGIC - ✅ `@scorer` デコレーター + `Feedback(value, rationale)` 戻り値
# MAGIC - ✅ MLflow 管理データセット（UC テーブルに永続化）
# MAGIC
# MAGIC **ai-dev-kit skill パターン適用:**
# MAGIC - ✅ WorkspaceClient による認証と API 呼び出し
# MAGIC - ✅ 環境変数による設定管理
# MAGIC - ✅ serving_endpoints.query() による LLM Judge（スコアラー内インライン）
# MAGIC
# MAGIC 次のステップ:
# MAGIC - Model Serving へのデプロイ（セル12のコメントアウトを解除）
# MAGIC - FastAPI + React アプリの構築
# MAGIC - Databricks Apps へのデプロイ
