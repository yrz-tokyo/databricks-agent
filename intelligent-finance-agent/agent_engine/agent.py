"""
Intelligent Finance Agent - Agent Implementation

このファイルは Databricks Agent Framework に準拠した Finance Agent の実装です。
- RAG Agent: Vector Search から IR 資料を検索し回答生成
- Insight Agent: RAG の回答を分析してインサイト生成
- LangGraph: 2つのエージェントを順次実行

開発ガイドライン (ai-dev-kit/databricks-python-sdk skill):
- 環境変数による設定管理 (os.environ.get())
- Databricks SDK の使用 (WorkspaceClient は databricks_langchain 内部で使用)
- ChatDatabricks による LLM 呼び出し
- MLflow ChatAgent パターンの実装
"""

from typing import Any, Generator, Optional, List, Dict
import os

import mlflow
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

############################################
# 環境変数から設定を取得
############################################
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

    config = {
        "llm_endpoint": llm_endpoint,
        "vs_name": vs_name
    }

    return config

# 設定を取得
config = get_env_config()

LLM_ENDPOINT_NAME = config['llm_endpoint']
VS_NAME = config['vs_name']

# 設定確認用の出力
print("=" * 60)
print("Finance Agent 設定:")
print(f"  LLM Endpoint: {LLM_ENDPOINT_NAME}")
print(f"  Vector Search: {VS_NAME}")
print("=" * 60)

# LangChain/MLflowの自動ロギングを有効化
mlflow.langchain.autolog()

############################################
# LLMインスタンスの作成
############################################
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# システムプロンプト（RAG エージェント用）
system_prompt = """あなたは企業IR資料の専門家です。提供されたツールを使用して、
質問に対して正確で詳細な回答を提供してください。

重要:
1. 数値を回答する際は、必ず単位を明記してください（百万円、億円など）
2. 該当ページ番号を引用してください
3. 複数の情報源がある場合は全て参照してください

まず、search_product_docs ツールで関連情報を検索してから回答してください。"""

############################################
# ツールの作成
############################################
def create_tools() -> List[BaseTool]:
    """Vector Search ツールとカスタムツールを作成"""
    tools = []

    # Vector Search ツールの追加（RAG用）
    if VS_NAME:
        try:
            vs_tool = VectorSearchRetrieverTool(
                index_name=VS_NAME,
                tool_name="search_product_docs",
                num_results=5,  # 5件の文書を取得
                tool_description="このツールを使用してIR資料から関連情報を検索します。売上、利益、事業セグメント、財務情報などの質問に対して使用してください。",
                disable_notice=True
            )
            tools.append(vs_tool)
            print(f"✓ Vector Search ツールを追加: {VS_NAME}")
        except Exception as e:
            print(f"Warning: Vector Search ツール {VS_NAME} をロードできませんでした: {e}")

    # インサイト生成ツール（カスタム）
    @tool
    def generate_insight(rag_answer: str, question: str) -> str:
        """
        RAG の回答を分析し、経営インサイトを生成します。

        Args:
            rag_answer: RAG Agent の回答
            question: 元の質問

        Returns:
            経営インサイト
        """
        insight_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは経営コンサルタントです。以下の質問と回答を分析し、
経営に役立つインサイトを提供してください。

以下の観点でインサイトを生成してください:

1. **主要なトレンドと変化**
   - データから読み取れる傾向
   - 前年比や成長率の分析

2. **潜在的なリスクと機会**
   - 懸念事項やリスク要因
   - 成長の機会や強み

3. **実行可能な提言**
   - 具体的なアクションアイテム
   - 優先順位の高い施策

必ず具体的な数値やファクトを引用し、根拠のあるインサイトを提供してください。"""),
            ("human", "質問: {question}\n\n回答: {rag_answer}\n\n上記の情報を分析し、経営インサイトを生成してください。")
        ])

        # Insight 用 LLM（temperature を少し高めに）
        insight_llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, temperature=0.3)

        try:
            messages = insight_prompt.format_messages(question=question, rag_answer=rag_answer)
            response = insight_llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"インサイト生成中にエラーが発生しました: {str(e)}"

    tools.append(generate_insight)
    print(f"✓ Insight 生成ツールを追加")

    return tools

# ツールを作成
tools = create_tools()

############################################
# Tool Calling Agent の作成
############################################
def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
):
    """ツール呼び出し型エージェントを作成"""

    # モデルにツールをバインド
    model = model.bind_tools(tools)

    # 次にどのノードに進むかを決定する関数
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # 関数呼び出しがあれば継続、なければ終了
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    # システムプロンプトを先頭に付与する前処理
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])

    model_runnable = preprocessor | model

    # モデル呼び出し用の関数
    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    # カスタムツール実行関数
    def execute_tools(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]

        # tool_callsを取得
        tool_calls = last_message.get("tool_calls", [])
        if not tool_calls:
            return {"messages": []}

        # ツールを実行
        tool_outputs = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name") if isinstance(tool_call, dict) else tool_call.function.name
            tool_args = tool_call.get("function", {}).get("arguments") if isinstance(tool_call, dict) else tool_call.function.arguments
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id

            # ツールを見つけて実行
            tool_result = None
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        # 引数をパース
                        import json
                        if isinstance(tool_args, str):
                            args = json.loads(tool_args)
                        else:
                            args = tool_args

                        # ツールを実行
                        result = tool.invoke(args)
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
                    break

            if tool_result is None:
                tool_result = f"Tool {tool_name} not found"

            # ツール実行結果のメッセージを作成
            tool_message = {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_id,
                "name": tool_name
            }
            tool_outputs.append(tool_message)

        return {"messages": tool_outputs}

    # LangGraphのワークフローを構築
    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()

############################################
# LangGraphChatAgent クラス（MLflow推論用ラッパー）
############################################
class LangGraphChatAgent(ChatAgent):
    """
    LangGraph の StateGraph を MLflow ChatAgent として使用するラッパー
    Databricks Agent Framework 標準
    """

    def __init__(self, agent):
        self.agent = agent

    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]) -> list[dict]:
        """ChatAgentMessage を辞書形式に変換"""
        converted = []

        if not messages:
            return converted

        for msg in messages:
            try:
                if msg is None:
                    print("Warning: None message encountered")
                    continue

                # ChatAgentMessageオブジェクトを辞書に変換
                if hasattr(msg, 'dict'):
                    msg_dict = msg.dict()
                elif isinstance(msg, dict):
                    msg_dict = msg
                else:
                    print(f"Warning: Unexpected message type: {type(msg)}")
                    continue

                # toolロールのメッセージの場合、contentが空の場合は処理
                if msg_dict.get("role") == "tool":
                    if not msg_dict.get("content"):
                        msg_dict["content"] = "Tool execution completed"

                    if "tool_call_id" not in msg_dict and msg_dict.get("id"):
                        msg_dict["tool_call_id"] = msg_dict["id"]

                converted.append(msg_dict)
            except Exception as e:
                print(f"Error converting message: {e}, Message: {msg}")
                continue

        return converted

    def predict(
        self,
        messages_or_context,
        context_or_model_input=None,
        custom_inputs_or_params=None,
    ) -> ChatAgentResponse:
        """
        同期的な予測メソッド（必須）

        mlflow.pyfunc は predict(context, model_input) として呼び出す場合と
        ChatAgent として predict(messages, context, custom_inputs) として呼び出す
        場合の両方を処理する。
        """
        # PythonModel 呼び出し規約 (context, model_input) の検出
        try:
            from mlflow.pyfunc import PythonModelContext
            if isinstance(messages_or_context, PythonModelContext):
                # pyfunc serving: predict(context, model_input)
                model_input = context_or_model_input
                if isinstance(model_input, dict):
                    raw = model_input.get("messages", [])
                elif hasattr(model_input, "to_dict"):
                    raw = model_input.to_dict(orient="records")[0].get("messages", [])
                else:
                    raw = []
                messages = [
                    ChatAgentMessage(**m) if isinstance(m, dict) else m
                    for m in raw
                ]
            else:
                # ChatAgent 呼び出し規約: predict(messages, context, custom_inputs)
                messages = messages_or_context or []
        except Exception:
            messages = messages_or_context or []

        return self._run_predict(messages)

    def _run_predict(self, messages: list) -> ChatAgentResponse:
        """LangGraph を実行してレスポンスを生成"""
        request = {"messages": self._convert_messages_to_dict(messages)}

        response_messages = []

        for event in self.agent.stream(request, stream_mode="updates"):
            if event and isinstance(event, dict):
                for node_data in event.values():
                    if node_data and isinstance(node_data, dict) and "messages" in node_data:
                        for msg in node_data.get("messages", []):
                            if msg is None:
                                continue

                            if hasattr(msg, 'dict'):
                                msg_dict = msg.dict()
                            elif isinstance(msg, dict):
                                msg_dict = msg
                            else:
                                continue

                            if msg_dict.get("role") == "tool":
                                if not msg_dict.get("content") and not msg_dict.get("tool_calls"):
                                    msg_dict["content"] = "Tool executed successfully"
                                if "tool_call_id" not in msg_dict and "id" in msg_dict:
                                    msg_dict["tool_call_id"] = msg_dict["id"]

                            try:
                                response_messages.append(ChatAgentMessage(**msg_dict))
                            except Exception as e:
                                print(f"Warning: Failed to create ChatAgentMessage: {e}, data: {msg_dict}")

        return ChatAgentResponse(messages=response_messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """ストリーミング予測メソッド"""
        request = {"messages": self._convert_messages_to_dict(messages)}

        for event in self.agent.stream(request, stream_mode="updates"):
            if event and isinstance(event, dict):
                for node_data in event.values():
                    if node_data and isinstance(node_data, dict) and "messages" in node_data:
                        for msg in node_data.get("messages", []):
                            if msg is None:
                                continue

                            if hasattr(msg, 'dict'):
                                msg_dict = msg.dict()
                            elif isinstance(msg, dict):
                                msg_dict = msg
                            else:
                                continue

                            if msg_dict.get("role") == "tool":
                                if not msg_dict.get("content") and not msg_dict.get("tool_calls"):
                                    msg_dict["content"] = "Tool executed successfully"
                                if "tool_call_id" not in msg_dict and "id" in msg_dict:
                                    msg_dict["tool_call_id"] = msg_dict["id"]

                            try:
                                yield ChatAgentChunk(**{"delta": msg_dict})
                            except Exception as e:
                                print(f"Warning: Failed to create ChatAgentChunk: {e}")

############################################
# 互換性ラッパー（開発・評価用）
############################################
class FinanceAgentWrapper:
    """
    開発・評価用の互換性ラッパー
    辞書形式の入力を受け付けて、LangGraphChatAgent に変換
    """

    def __init__(self, agent: LangGraphChatAgent):
        self.agent = agent

    def predict(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        辞書形式の入力を受け付けて実行

        Args:
            model_input: {"messages": [{"role": "user", "content": "..."}]}

        Returns:
            {"messages": [...], "question": "...", ...}
        """
        # 辞書を ChatAgentMessage に変換
        messages = []
        for msg_dict in model_input.get("messages", []):
            messages.append(ChatAgentMessage(**msg_dict))

        # Agent を実行
        response = self.agent.predict(messages)

        # 結果を辞書形式に変換
        result_messages = []
        for msg in response.messages:
            if hasattr(msg, 'dict'):
                result_messages.append(msg.dict())
            elif isinstance(msg, dict):
                result_messages.append(msg)

        # 質問を抽出
        question = ""
        if model_input.get("messages"):
            last_msg = model_input["messages"][-1]
            question = last_msg.get("content", "")

        return {
            "messages": result_messages,
            "question": question
        }

############################################
# Agent の作成と登録
############################################
# エージェントオブジェクトを作成し、mlflow.models.set_model()で推論時に使用するエージェントとして指定
agent_graph = create_tool_calling_agent(llm, tools, system_prompt)
base_agent = LangGraphChatAgent(agent_graph)

# MLflow serving 用には base_agent (ChatAgent) を登録
# mlflow.pyfunc.log_model で ChatAgent として正しく扱われる
# - Playground 対応: ChatAgent の chat completions スキーマを認識
# - ストリーミング対応: predict_stream() が自動的に呼ばれる
mlflow.models.set_model(base_agent)

# 開発・評価用には互換性ラッパーを提供
AGENT = FinanceAgentWrapper(base_agent)

print("=" * 60)
print("✓ Finance Agent 作成完了")
print("  - LangGraph StateGraph を構築")
print("  - LangGraphChatAgent でラップ")
print("  - FinanceAgentWrapper で互換性確保")
print("  - mlflow.models.set_model(base_agent) 完了")
print("=" * 60)
