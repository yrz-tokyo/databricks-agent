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
# 環境変数から設定を取得（遅延評価）
############################################
def get_llm_endpoint() -> str:
    val = os.environ.get("LLM_ENDPOINT_NAME")
    if not val:
        raise ValueError("LLM_ENDPOINT_NAME environment variable is required")
    return val

def get_vs_name() -> str:
    val = os.environ.get("VS_NAME")
    if not val:
        raise ValueError("VS_NAME environment variable is required")
    return val

# システムプロンプト（RAG エージェント用）
system_prompt = """あなたは企業IR資料の専門家です。提供されたツールを使用して、
質問に対して正確で詳細な回答を提供してください。

重要:
1. 数値を回答する際は、必ず単位を明記してください（百万円、億円など）
2. 該当ページ番号を引用してください
3. 複数の情報源がある場合は全て参照してください

まず、search_product_docs ツールで関連情報を検索してから回答してください。"""

############################################
# ツールの作成（リクエスト時に初期化）
############################################
def create_tools() -> List[BaseTool]:
    """Vector Search ツールとカスタムツールを作成"""
    tools = []
    vs_name = get_vs_name()
    llm_endpoint = get_llm_endpoint()

    # Vector Search ツールの追加（RAG用）
    try:
        vs_tool = VectorSearchRetrieverTool(
            index_name=vs_name,
            tool_name="search_product_docs",
            num_results=5,
            tool_description="このツールを使用してIR資料から関連情報を検索します。売上、利益、事業セグメント、財務情報などの質問に対して使用してください。",
            disable_notice=True
        )
        tools.append(vs_tool)
        print(f"✓ Vector Search ツールを追加: {vs_name}")
    except Exception as e:
        print(f"Warning: Vector Search ツール {vs_name} をロードできませんでした: {e}")

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

        insight_llm = ChatDatabricks(endpoint=os.environ.get("LLM_ENDPOINT_NAME"), temperature=0.3)

        try:
            messages = insight_prompt.format_messages(question=question, rag_answer=rag_answer)
            response = insight_llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"インサイト生成中にエラーが発生しました: {str(e)}"

    tools.append(generate_insight)
    print(f"✓ Insight 生成ツールを追加")

    return tools

############################################
# Tool Calling Agent の作成
############################################
def create_agent_graph(system_prompt: Optional[str] = None):
    """環境変数が利用可能になってからLLMとツールを初期化してグラフを作成"""
    llm = ChatDatabricks(endpoint=get_llm_endpoint())
    tools = create_tools()
    return create_tool_calling_agent(llm, tools, system_prompt)


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
    エージェントグラフは初回リクエスト時に遅延初期化する。
    """

    def __init__(self):
        self._agent = None

    @property
    def agent(self):
        if self._agent is None:
            print("Lazy initializing agent graph...")
            self._agent = create_agent_graph(system_prompt)
            print("✓ Agent graph initialized")
        return self._agent

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
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """
        同期的な予測メソッド（必須）

        Args:
            messages: 入力メッセージのリスト
            context: オプションのコンテキスト
            custom_inputs: カスタム入力

        Returns:
            ChatAgentResponse オブジェクト
        """
        # 入力メッセージを辞書形式に変換
        request = {"messages": self._convert_messages_to_dict(messages)}

        response_messages = []

        # LangGraphのストリームからメッセージを収集
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue

                                # メッセージオブジェクトを辞書に変換
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif isinstance(msg, dict):
                                    msg_dict = msg
                                else:
                                    print(f"Warning: Unexpected message type: {type(msg)}")
                                    continue

                                # toolメッセージの内容を確認
                                if msg_dict.get("role") == "tool":
                                    if not msg_dict.get("content") and not msg_dict.get("tool_calls"):
                                        msg_dict["content"] = "Tool executed successfully"

                                    if "tool_call_id" not in msg_dict and "id" in msg_dict:
                                        msg_dict["tool_call_id"] = msg_dict["id"]

                                try:
                                    response_messages.append(ChatAgentMessage(**msg_dict))
                                except Exception as e:
                                    print(f"Warning: Failed to create ChatAgentMessage: {e}")
                                    print(f"Message data: {msg_dict}")

        except Exception as e:
            print(f"Error in predict method: {e}")
            import traceback
            traceback.print_exc()

        return ChatAgentResponse(messages=response_messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """
        ストリーミング予測メソッド（推奨）

        Args:
            messages: 入力メッセージのリスト
            context: オプションのコンテキスト
            custom_inputs: カスタム入力

        Yields:
            ChatAgentChunk オブジェクト
        """
        # 入力メッセージを辞書形式に変換
        request = {"messages": self._convert_messages_to_dict(messages)}

        # ストリームで逐次応答を生成
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue

                                # メッセージオブジェクトを辞書に変換
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif isinstance(msg, dict):
                                    msg_dict = msg
                                else:
                                    print(f"Warning: Unexpected message type in stream: {type(msg)}")
                                    continue

                                # toolメッセージの内容を確認
                                if msg_dict.get("role") == "tool":
                                    if not msg_dict.get("content") and not msg_dict.get("tool_calls"):
                                        msg_dict["content"] = "Tool executed successfully"

                                    if "tool_call_id" not in msg_dict and "id" in msg_dict:
                                        msg_dict["tool_call_id"] = msg_dict["id"]

                                try:
                                    yield ChatAgentChunk(**{"delta": msg_dict})
                                except Exception as e:
                                    print(f"Warning: Failed to create ChatAgentChunk: {e}")
                                    continue

        except Exception as e:
            print(f"Error in predict_stream method: {e}")
            import traceback
            traceback.print_exc()
            return

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
# LangChain/MLflowの自動ロギングを有効化
mlflow.langchain.autolog()

# エージェントはlazy initializationで起動時の接続を回避
base_agent = LangGraphChatAgent()

# MLflow 用には base_agent を登録
mlflow.models.set_model(base_agent)

# 開発・評価用には互換性ラッパーを提供
AGENT = FinanceAgentWrapper(base_agent)

print("=" * 60)
print("✓ Finance Agent 登録完了 (lazy initialization)")
print("  - LangGraphChatAgent 作成（初回リクエスト時に初期化）")
print("  - FinanceAgentWrapper で互換性確保")
print("  - mlflow.models.set_model() 完了")
print("=" * 60)
