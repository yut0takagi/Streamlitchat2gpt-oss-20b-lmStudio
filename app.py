# app.py
# Streamlit-based ChatGPT-like UI that talks to a local LM Studio server
# via the OpenAI-compatible Chat Completions API.
#
# Requirements:
#   pip install -r requirements.txt
#
# Usage:
#   1) In LM Studio: Developer > Local Server > Start Server
#      - Load model: openai/gpt-oss-20b (or your preferred model)
#      - Confirm the endpoint (default http://localhost:1234/v1)
#   2) Run this app:
#      streamlit run app.py
#
# Notes:
#   - 100% local inference when LM Studio runs on your machine.
#   - Supports streaming, system prompt, and basic sampling controls.
#   - Conversation history is kept in Streamlit's session state.
#
import time
import json
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# Use OpenAI Python SDK (works with LM Studio in OpenAI-compat mode)
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None

# Serp API for web search
try:
    from serpapi import GoogleSearch
except Exception as e:
    GoogleSearch = None

APP_TITLE = "Local Chat (LM Studio + Chat Completions)"

# -----------------------------
# .env loader and configuration
# -----------------------------
def _load_env_file(filepath: str = ".env") -> dict:
    env: dict = {}
    p = Path(filepath)
    if not p.exists():
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        env[k] = v
    for k, v in env.items():
        if k not in os.environ:
            os.environ[k] = v
    return env

_ENV_CACHE = _load_env_file()

def _get_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default

DEFAULT_BASE_URL = _get_env("OPENAI_BASE_URL", "BASE_URL", "LMSTUDIO_BASE_URL", default="http://localhost:1234/v1")
DEFAULT_API_KEY = _get_env("OPENAI_API_KEY", "API_KEY", default="lm-studio")
DEFAULT_MODEL = _get_env("OPENAI_MODEL", "MODEL", default="openai/gpt-oss-20b")
SERPAPI_KEY = _get_env("SERPAPI_KEY", default=None)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# -----------------------------
# Session state initialization (threads + persistence)
# -----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def _new_thread(title: str = "新規スレッド") -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "system": "あなたは有能なアシスタントです。簡潔かつ正確に回答してください。",
        "messages": [],
    }

def _thread_path(thread_id: str) -> Path:
    return DATA_DIR / f"thread_{thread_id}.json"

def _save_threads_index():
    idx = {
        "threads": [
            {"id": t["id"], "title": t.get("title") or "無題", "updated_at": t.get("updated_at"), "path": str(_thread_path(t["id"]))}
            for t in st.session_state.threads
        ]
    }
    (DATA_DIR / "threads_index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

def _save_thread(thread: Dict[str, Any]):
    thread["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _thread_path(thread["id"]).write_text(json.dumps(thread, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_threads_index()

def _load_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    p = _thread_path(thread_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_threads_index() -> List[Dict[str, Any]]:
    idx_path = DATA_DIR / "threads_index.json"
    if not idx_path.exists():
        return []
    try:
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        threads: List[Dict[str, Any]] = []
        for item in data.get("threads", []):
            tid = item.get("id")
            if not tid:
                continue
            t = _load_thread(tid)
            if t:
                threads.append(t)
        return threads
    except Exception:
        return []

if "threads" not in st.session_state:
    loaded_threads = _load_threads_index()
    st.session_state.threads: List[Dict[str, Any]] = loaded_threads if loaded_threads else [_new_thread()]

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = st.session_state.threads[0]["id"]

if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""

if "stop_now" not in st.session_state:
    st.session_state.stop_now = False

if "enable_search" not in st.session_state:
    st.session_state.enable_search = False

if "num_search_results" not in st.session_state:
    st.session_state.num_search_results = 5

def _get_current_thread() -> Dict[str, Any]:
    tid = st.session_state.current_thread_id
    for t in st.session_state.threads:
        if t["id"] == tid:
            return t
    st.session_state.current_thread_id = st.session_state.threads[0]["id"]
    return st.session_state.threads[0]

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("チャットスレッド")
    thread_titles = [t.get("title") or "無題" for t in st.session_state.threads]
    thread_ids = [t["id"] for t in st.session_state.threads]
    if thread_ids:
        current_index = thread_ids.index(st.session_state.current_thread_id) if st.session_state.current_thread_id in thread_ids else 0
        sel = st.radio("スレッドを選択", options=list(range(len(thread_ids))), index=current_index, format_func=lambda i: thread_titles[i], key="thread_selector")
        st.session_state.current_thread_id = thread_ids[sel]
        # title editor
        cur = _get_current_thread()
        new_title_val = st.text_input("スレッド名", value=cur.get("title", "無題"), key=f"title_{cur['id']}")
        if new_title_val != cur.get("title"):
            cur["title"] = new_title_val
            _save_thread(cur)
    c1, c3, c4 = st.columns([1,1,1])
    with c1:
        if st.button("新規"):
            t = _new_thread()
            st.session_state.threads.insert(0, t)
            st.session_state.current_thread_id = t["id"]
            _save_thread(t)
            st.rerun()
    with c3:
        if st.button("削除"):
            cur = _get_current_thread()
            try:
                _thread_path(cur["id"]).unlink(missing_ok=True)
            except Exception:
                pass
            st.session_state.threads = [t for t in st.session_state.threads if t["id"] != cur["id"]]
            if not st.session_state.threads:
                st.session_state.threads = [_new_thread()]
            st.session_state.current_thread_id = st.session_state.threads[0]["id"]
            _save_threads_index()
            st.rerun()
    with c4:
        if st.button("クリア"):
            cur = _get_current_thread()
            cur["messages"] = []
            st.session_state.last_user_input = ""
            st.session_state.stop_now = False
            _save_thread(cur)
            st.success("このスレッドの履歴をクリアしました")
    st.divider()
    st.subheader("接続設定 (.env から読込)")
    base_url = DEFAULT_BASE_URL
    api_key = DEFAULT_API_KEY
    model_name = DEFAULT_MODEL
    st.text_input("Base URL", value=base_url, help="LM StudioのOpenAI互換API", disabled=True)
    st.text_input("API Key（任意）", value=api_key, type="password", help="LM Studioでは任意文字列でOK", disabled=True)
    st.text_input("Model ID", value=model_name, help="例: openai/gpt-oss-20b", disabled=True)
    if st.button(".env を再読み込み"):
        _load_env_file()
        st.toast(".env を再読み込みしました")
        st.rerun()
    colA, colB = st.columns(2)
    with colA:
        if st.button("接続テスト / モデル一覧取得"):
            try:
                if OpenAI is None:
                    st.error("openai パッケージの import に失敗しました。`pip install openai` を実行してください。")
                else:
                    client = OpenAI(base_url=base_url, api_key=api_key)
                    models = client.models.list()
                    ids = [m.id for m in models.data]
                    st.success("接続OK")
                    st.write("使用可能モデル:", ids)
            except Exception as e:
                st.error(f"接続エラー: {e}")
    with colB:
        if st.button("このスレッドをクリア"):
            cur = _get_current_thread()
            cur["messages"] = []
            st.session_state.last_user_input = ""
            st.session_state.stop_now = False
            _save_thread(cur)
            st.success("このスレッドの履歴をクリアしました")

    st.divider()
    st.subheader("サンプリング設定")
    temperature = st.slider("temperature", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("top_p", 0.0, 1.0, 1.0, 0.05)
    presence_penalty = st.slider("presence_penalty", -2.0, 2.0, 0.0, 0.1)
    frequency_penalty = st.slider("frequency_penalty", -2.0, 2.0, 0.0, 0.1)
    max_tokens_opt = st.number_input("max_tokens (-1で制限なし)", value=-1, step=64, min_value=-1)
    stop_sequences_raw = st.text_area("stop（改行区切り、任意）", value="")
    stop_sequences = [s for s in (stop_sequences_raw.splitlines()) if s.strip()] or None
    ctx_limit = st.number_input("過去ラウンドの保持数（直近N往復）", min_value=1, max_value=50, value=12, step=1,
                                help="長文になりがちな場合は小さめに。systemは常に先頭に付与します。")

    st.divider()
    st.subheader("検索設定（Serp API）")
    st.session_state.enable_search = st.toggle("検索を有効化", value=st.session_state.enable_search, help="ユーザー入力に対して検索を実行し、結果をプロンプトに含めます")
    if st.session_state.enable_search:
        if not SERPAPI_KEY:
            st.warning("⚠️ SERPAPI_KEYが設定されていません。.envファイルに追加してください。")
        st.session_state.num_search_results = st.number_input("検索結果数", min_value=1, max_value=10, value=st.session_state.num_search_results, step=1)

    st.divider()
    st.subheader("System Prompt")
    _cur = _get_current_thread()
    _cur["system"] = st.text_area(
        "system", value=_cur.get("system", ""), height=140,
        help="このプロンプトは最初のsystemメッセージとして付与されます。"
    )
    _save_thread(_cur)

    st.divider()
    st.caption("このアプリはLM StudioのOpenAI互換APIに接続し、Chat Completionsで応答を生成します。")


def _search_with_serpapi(query: str, num_results: int = 5) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Serp APIを使用して検索を実行し、結果をフォーマットして返す
    Returns: (formatted_results_text, search_metadata)
    """
    if not SERPAPI_KEY:
        return None, {"error": "SERPAPI_KEYが設定されていません"}
    if GoogleSearch is None:
        return None, {"error": "google-search-resultsパッケージがインストールされていません"}
    
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "hl": "ja",
            "gl": "jp",
            "num": num_results
        })
        results = search.get_dict()
        
        # デバッグ: レスポンスのキーを確認
        response_keys = list(results.keys())
        
        # 検索結果をフォーマット
        formatted_results = []
        
        # organic_results（通常の検索結果）
        organic_results = results.get("organic_results", [])
        result_count = len(organic_results)
        
        for i, result in enumerate(organic_results[:num_results], 1):
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            if title or link or snippet:
                formatted_results.append(f"検索結果 {i}:\nタイトル: {title}\nURL: {link}\n内容: {snippet}\n")
        
        # knowledge_graph（知識グラフ）がある場合
        knowledge_graph = results.get("knowledge_graph", {})
        if knowledge_graph:
            kg_title = knowledge_graph.get("title", "")
            kg_type = knowledge_graph.get("type", "")
            kg_description = knowledge_graph.get("description", "")
            kg_source = knowledge_graph.get("source", {})
            kg_source_name = kg_source.get("name", "") if isinstance(kg_source, dict) else ""
            kg_source_link = kg_source.get("link", "") if isinstance(kg_source, dict) else ""
            
            if kg_title or kg_description:
                formatted_results.insert(0, f"【知識グラフ - 信頼性の高い情報】\nタイトル: {kg_title}\n種類: {kg_type}\n説明: {kg_description}\n出典: {kg_source_name} ({kg_source_link})\n\n")
        
        # answer_box（回答ボックス）がある場合
        answer_box = results.get("answer_box", {})
        if answer_box:
            answer_title = answer_box.get("title", "")
            answer_answer = answer_box.get("answer", "")
            answer_snippet = answer_box.get("snippet", "")
            answer_link = answer_box.get("link", "")
            
            if answer_title or answer_answer or answer_snippet:
                formatted_results.insert(0, f"【直接回答 - 最も関連性の高い情報】\n質問: {answer_title}\n回答: {answer_answer}\n詳細: {answer_snippet}\n参考リンク: {answer_link}\n\n")
        
        # local_results（ローカル検索結果）がある場合
        local_results = results.get("local_results", [])
        if local_results:
            # local_resultsがリストかどうかを確認
            if isinstance(local_results, list):
                for i, local in enumerate(local_results[:3], 1):
                    if isinstance(local, dict):
                        local_title = local.get("title", "")
                        local_address = local.get("address", "")
                        local_phone = local.get("phone", "")
                        local_website = local.get("website", "")
                        if local_title:
                            formatted_results.append(f"【ローカル結果 {i}】\n{local_title}\n住所: {local_address}\n電話: {local_phone}\nウェブサイト: {local_website}\n\n")
            elif isinstance(local_results, dict):
                # 辞書型の場合（単一のローカル結果）
                local_title = local_results.get("title", "")
                local_address = local_results.get("address", "")
                local_phone = local_results.get("phone", "")
                local_website = local_results.get("website", "")
                if local_title:
                    formatted_results.append(f"【ローカル結果】\n{local_title}\n住所: {local_address}\n電話: {local_phone}\nウェブサイト: {local_website}\n\n")
        
        metadata = {
            "query": query,
            "result_count": result_count,
            "requested_count": num_results,
            "success": True,
            "response_keys": response_keys,
            "has_knowledge_graph": bool(knowledge_graph),
            "has_answer_box": bool(answer_box),
            "has_local_results": bool(local_results)
        }
        
        if formatted_results:
            formatted_text = "=== 最新の検索結果（質問に回答する際はこの情報を優先的に使用してください） ===\n\n" + "\n".join(formatted_results) + "\n=== 検索結果終了 ===\n"
            return formatted_text, metadata
        else:
            # デバッグ情報を含める
            metadata["error"] = "検索結果が見つかりませんでした"
            metadata["debug_info"] = f"レスポンスキー: {response_keys}, organic_results数: {result_count}"
            return None, metadata
    except Exception as e:
        error_msg = str(e)
        import traceback
        return None, {"error": f"検索エラー: {error_msg}", "query": query, "success": False, "traceback": traceback.format_exc()}


def _build_payload_messages(include_search: bool = False, search_query: Optional[str] = None) -> Tuple[List[Dict[str, str]], Optional[str], Optional[Dict[str, Any]]]:
    """Assemble messages including system prompt and clipped history for current thread.
    Returns: (payload_messages, search_results_text, search_metadata)
    """
    t = _get_current_thread()
    history = t["messages"][-(ctx_limit*2):]
    system_content = t.get("system", "")
    search_results_text = None
    search_metadata = None
    
    # 検索結果を含める場合
    if include_search and search_query:
        search_results_text, search_metadata = _search_with_serpapi(search_query, st.session_state.num_search_results)
    
    # Systemプロンプトに検索結果の活用方法を追加
    if include_search and search_results_text:
        if "検索結果" not in system_content and "以下の検索結果" not in system_content:
            system_content = f"{system_content}\n\n【重要】検索機能が有効です。ユーザーの質問に対して、提供された検索結果を必ず参照して回答してください。検索結果に含まれる最新の情報を優先的に使用し、検索結果に基づいた正確で詳細な回答を提供してください。検索結果にない情報は推測せず、検索結果の内容を明確に引用してください。"
    
    payload = [{"role": "system", "content": system_content}]
    
    # 履歴メッセージを追加（検索結果は最後のuserメッセージの直前に挿入）
    for idx, m in enumerate(history):
        role = m.get("role")
        content = m.get("content", "")
        
        # 最後のuserメッセージの直前に検索結果を挿入
        is_last_user_message = (role == "user" and idx == len(history) - 1)
        if is_last_user_message and include_search and search_results_text:
            # 検索結果をuserメッセージの前に追加（より明確な構造で）
            search_context = f"{search_results_text}\n\n---\n\n上記の検索結果を参照して、以下の質問に回答してください:\n\n{content}"
            payload.append({"role": "user", "content": search_context})
        else:
            payload.append({"role": role, "content": content})
    
    return payload, search_results_text, search_metadata


def _ensure_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        st.error("openai パッケージが見つかりません。`pip install openai` を先に実行してください。")
        return None
    try:
        return OpenAI(base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY)
    except Exception as e:
        st.error(f"OpenAIクライアント初期化エラー: {e}")
        return None


def _chat_once_streaming(user_input: str):
    """Send a user_input and stream the assistant response into the UI."""
    client = _ensure_client()
    if client is None:
        return

    # append user message to current thread and persist
    thread = _get_current_thread()
    thread["messages"].append({"role": "user", "content": user_input, "ts": time.time()})
    _save_thread(thread)
    st.session_state.last_user_input = user_input

    # render past messages (including the new user message)
    for m in thread["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # placeholder for the streaming assistant message
    with st.chat_message("assistant"):
        placeholder = st.empty()

    # 検索を実行（有効な場合）
    search_query = user_input if st.session_state.enable_search else None
    search_status_container = st.empty()
    
    if st.session_state.enable_search and search_query:
        with search_status_container.container():
            with st.status("🔍 検索を実行中...", expanded=True) as status:
                st.write(f"検索クエリ: **{search_query}**")
                payload_messages, search_results_text, search_metadata = _build_payload_messages(include_search=True, search_query=search_query)
                
                if search_metadata:
                    if search_metadata.get("success"):
                        status.update(label="✅ 検索完了", state="complete")
                        st.success(f"検索結果を {search_metadata.get('result_count', 0)} 件取得しました")
                    else:
                        status.update(label="❌ 検索エラー", state="error")
                        error_msg = search_metadata.get("error", "不明なエラー")
                        st.error(f"検索に失敗しました: {error_msg}")
                else:
                    status.update(label="⚠️ 検索未実行", state="error")
                    st.warning("検索が実行されませんでした")
    else:
        payload_messages, search_results_text, search_metadata = _build_payload_messages(include_search=False, search_query=None)
        if not st.session_state.enable_search:
            with search_status_container.container():
                st.info("ℹ️ 検索機能は無効です。サイドバーで有効化できます。")
    
    # 検索結果を表示（詳細）
    if search_results_text and search_metadata and search_metadata.get("success"):
        with st.expander("🔍 検索結果の詳細", expanded=True):
            st.markdown(f"**検索クエリ:** {search_metadata.get('query', 'N/A')}")
            st.markdown(f"**取得件数:** {search_metadata.get('result_count', 0)} / {search_metadata.get('requested_count', 0)} 件")
            if search_metadata.get("has_knowledge_graph"):
                st.markdown("✅ 知識グラフあり")
            if search_metadata.get("has_answer_box"):
                st.markdown("✅ 回答ボックスあり")
            if search_metadata.get("has_local_results"):
                st.markdown("✅ ローカル結果あり")
            st.divider()
            st.markdown(search_results_text)
    elif search_metadata:
        if search_metadata.get("success") and search_metadata.get("result_count", 0) == 0:
            # 検索は成功したが結果が0件の場合
            with st.expander("🔍 検索結果", expanded=True):
                st.warning(f"**検索結果が0件でした**")
                st.markdown(f"**検索クエリ:** {search_metadata.get('query', 'N/A')}")
                if search_metadata.get("debug_info"):
                    st.code(search_metadata.get("debug_info"), language="text")
                if search_metadata.get("response_keys"):
                    st.markdown(f"**レスポンスキー:** {', '.join(search_metadata.get('response_keys', []))}")
        elif not search_metadata.get("success"):
            # 検索エラーの場合
            with st.expander("🔍 検索結果", expanded=True):
                st.error(f"**エラー:** {search_metadata.get('error', '不明なエラー')}")
                if search_metadata.get("traceback"):
                    with st.expander("詳細なエラー情報", expanded=False):
                        st.code(search_metadata.get("traceback"), language="python")

    # streaming request
    started_at = time.time()
    full_text = ""
    st.session_state.stop_now = False
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=payload_messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=None if max_tokens_opt == -1 else int(max_tokens_opt),
            stream=True,
            stop=stop_sequences,
        )
        for chunk in completion:
            if st.session_state.stop_now:
                break
            if (hasattr(chunk, "choices")
                and chunk.choices
                and getattr(chunk.choices[0], "delta", None) is not None):
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    full_text += content
                    # live update
                    placeholder.markdown(full_text)
                    # persist partial assistant content
                    if thread["messages"] and thread["messages"][-1].get("role") == "assistant":
                        thread["messages"][-1]["content"] = full_text
                        thread["messages"][-1]["ts"] = time.time()
                    else:
                        thread["messages"].append({"role": "assistant", "content": full_text, "ts": time.time()})
                    _save_thread(thread)
    except Exception as e:
        placeholder.markdown(f":red[エラーが発生しました: {e}]")
        return

    elapsed = time.time() - started_at
    # finalize: ensure assistant message saved
    if thread["messages"] and thread["messages"][-1].get("role") == "assistant":
        thread["messages"][-1]["content"] = full_text
        thread["messages"][-1]["ts"] = time.time()
    else:
        thread["messages"].append({"role": "assistant", "content": full_text, "ts": time.time()})
    _save_thread(thread)

    # show meta
    with st.expander("生成メタ情報", expanded=False):
        st.write({
            "elapsed_seconds": round(elapsed, 3),
            "chars": len(full_text),
        })
        st.caption("Streaming時はトークン使用量が返らない場合があります。必要なら非Streamingモードに切り替えてください。")


def _chat_once_non_streaming(user_input: str):
    """Send a user_input and render full assistant response (no streaming)."""
    client = _ensure_client()
    if client is None:
        return

    thread = _get_current_thread()
    thread["messages"].append({"role": "user", "content": user_input, "ts": time.time()})
    _save_thread(thread)
    st.session_state.last_user_input = user_input

    for m in thread["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    with st.chat_message("assistant"):
        placeholder = st.empty()

    # 検索を実行（有効な場合）
    search_query = user_input if st.session_state.enable_search else None
    search_status_container = st.empty()
    
    if st.session_state.enable_search and search_query:
        with search_status_container.container():
            with st.status("🔍 検索を実行中...", expanded=True) as status:
                st.write(f"検索クエリ: **{search_query}**")
                payload_messages, search_results_text, search_metadata = _build_payload_messages(include_search=True, search_query=search_query)
                
                if search_metadata:
                    if search_metadata.get("success"):
                        status.update(label="✅ 検索完了", state="complete")
                        st.success(f"検索結果を {search_metadata.get('result_count', 0)} 件取得しました")
                    else:
                        status.update(label="❌ 検索エラー", state="error")
                        error_msg = search_metadata.get("error", "不明なエラー")
                        st.error(f"検索に失敗しました: {error_msg}")
                else:
                    status.update(label="⚠️ 検索未実行", state="error")
                    st.warning("検索が実行されませんでした")
    else:
        payload_messages, search_results_text, search_metadata = _build_payload_messages(include_search=False, search_query=None)
        if not st.session_state.enable_search:
            with search_status_container.container():
                st.info("ℹ️ 検索機能は無効です。サイドバーで有効化できます。")
    
    # 検索結果を表示（詳細）
    if search_results_text and search_metadata and search_metadata.get("success"):
        with st.expander("🔍 検索結果の詳細", expanded=True):
            st.markdown(f"**検索クエリ:** {search_metadata.get('query', 'N/A')}")
            st.markdown(f"**取得件数:** {search_metadata.get('result_count', 0)} / {search_metadata.get('requested_count', 0)} 件")
            if search_metadata.get("has_knowledge_graph"):
                st.markdown("✅ 知識グラフあり")
            if search_metadata.get("has_answer_box"):
                st.markdown("✅ 回答ボックスあり")
            if search_metadata.get("has_local_results"):
                st.markdown("✅ ローカル結果あり")
            st.divider()
            st.markdown(search_results_text)
    elif search_metadata:
        if search_metadata.get("success") and search_metadata.get("result_count", 0) == 0:
            # 検索は成功したが結果が0件の場合
            with st.expander("🔍 検索結果", expanded=True):
                st.warning(f"**検索結果が0件でした**")
                st.markdown(f"**検索クエリ:** {search_metadata.get('query', 'N/A')}")
                if search_metadata.get("debug_info"):
                    st.code(search_metadata.get("debug_info"), language="text")
                if search_metadata.get("response_keys"):
                    st.markdown(f"**レスポンスキー:** {', '.join(search_metadata.get('response_keys', []))}")
        elif not search_metadata.get("success"):
            # 検索エラーの場合
            with st.expander("🔍 検索結果", expanded=True):
                st.error(f"**エラー:** {search_metadata.get('error', '不明なエラー')}")
                if search_metadata.get("traceback"):
                    with st.expander("詳細なエラー情報", expanded=False):
                        st.code(search_metadata.get("traceback"), language="python")

    started_at = time.time()
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=payload_messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=None if max_tokens_opt == -1 else int(max_tokens_opt),
            stream=False,
            stop=stop_sequences,
        )
        content = completion.choices[0].message.content
        elapsed = time.time() - started_at
        usage = getattr(completion, "usage", None)
        thread["messages"].append({"role": "assistant", "content": content, "ts": time.time()})
        _save_thread(thread)
        placeholder.markdown(content)

        with st.expander("生成メタ情報", expanded=False):
            st.write({
                "elapsed_seconds": round(elapsed, 3),
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage.__dict__ if usage else None),
            })
    except Exception as e:
        placeholder.markdown(f":red[エラーが発生しました: {e}]")
        return


# -----------------------------
# Main chat area (history)
# -----------------------------
for m in _get_current_thread()["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Controls for streaming vs. non-streaming + stop/regenerate
col1, col2, col3 = st.columns([1,1,2])
with col1:
    use_streaming = st.toggle("Streamingで生成", value=True, help="トークン到着ごとに表示します")
with col2:
    if st.button("⏹ 停止", help="Streaming中に押すと中断します"):
        st.session_state.stop_now = True
with col3:
    if st.button("🔁 最後の質問を再生成"):
        if st.session_state.last_user_input:
            if use_streaming:
                _chat_once_streaming(st.session_state.last_user_input)
            else:
                _chat_once_non_streaming(st.session_state.last_user_input)
        else:
            st.toast("直近のユーザー入力が見つかりません", icon="⚠️")

# Chat input
user_input = st.chat_input("メッセージを入力（Shift+Enterで改行）")
if user_input:
    if use_streaming:
        _chat_once_streaming(user_input)
    else:
        _chat_once_non_streaming(user_input)

# Export / Import
with st.expander("エクスポート / インポート", expanded=False):
    cur = _get_current_thread()
    export_obj = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "system": cur.get("system", ""),
        "messages": cur["messages"],
    }
    st.download_button(
        "💾 履歴をJSONでダウンロード",
        data=json.dumps(export_obj, ensure_ascii=False, indent=2),
        file_name="chat_history.json",
        mime="application/json",
    )

    uploaded = st.file_uploader("JSONファイルから履歴を読み込み（上書き）", type=["json"])
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            cur = _get_current_thread()
            cur["system"] = data.get("system", cur.get("system"))
            cur["messages"] = data.get("messages", [])
            _save_thread(cur)
            st.success("読み込みました。画面を更新して反映。")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")
