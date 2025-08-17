# LM Studio ChatUI（OpenAI互換APIクライアント）

このリポジトリは、[LM Studio](https://lmstudio.ai/) のOpenAI互換API（Chat Completions）に接続し、StreamlitベースのチャットUIを提供します。  
ローカルで大規模言語モデル（LLM）を手軽に試したい方に最適です。

---

## 特徴

- **LM StudioのOpenAI互換API** に対応
- **Streamlit製のシンプルなUI**（Webブラウザで利用可能）
- **会話履歴の保存・エクスポート/インポート**（JSON形式）
- **System Promptや各種パラメータの調整**（temperature, top_p, max_tokens等）
- **Streaming表示対応**（トークン到着ごとに即時表示）

---

## 必要条件

- [LM Studio](https://lmstudio.ai/)（ローカルPCで起動し、OpenAI互換APIを有効化してください）
- Docker（推奨）またはPython 3.8以降

---

## クイックスタート（Docker推奨）

1. **LM Studioを起動し、OpenAI API互換サーバーを有効化**  
   - LM Studioの「API」タブで「OpenAI Compatible API」をONにし、`http://localhost:1234/v1` で待ち受けてください。

2. **このリポジトリをクローン**
   ```sh
   git clone https://github.com/yut0takagi/lmstudio-chatui.git
   cd lmstudio-chatui
   ```

3. **.envファイルを作成（必要に応じて）**  
   例:
   ```
   OPENAI_BASE_URL=http://host.docker.internal:1234/v1
   OPENAI_API_KEY=lm-studio
   OPENAI_MODEL=openai/gpt-oss-20b
   ```

4. **Dockerで起動**
   ```sh
   docker compose up --build
   ```
   - ブラウザで [http://localhost:8501](http://localhost:8501) を開いてください。

---

## ローカル（Python）での実行

1. 必要なパッケージをインストール
   ```sh
   pip install -r requirements.txt
   ```

2. `.env`ファイルを作成（上記参照）

3. アプリを起動
   ```sh
   streamlit run app.py
   ```

---

## 主要な環境変数

| 変数名             | 説明                                         | 例                                      |
|--------------------|----------------------------------------------|-----------------------------------------|
| OPENAI_BASE_URL    | LM StudioのAPIエンドポイント                 | http://host.docker.internal:1234/v1     |
| OPENAI_API_KEY     | 任意（LM Studioでは任意文字列でOK）           | lm-studio                               |
| OPENAI_MODEL       | 使用するモデル名（LM Studioで有効なもの）     | openai/gpt-oss-20b                      |

---

## よくある質問

- **Q. LM Studioが動いていないとエラーになりますか？**  
  A. はい。必ずLM Studioを起動し、APIサーバーを有効化してください。

- **Q. モデル名はどこで確認できますか？**  
  A. LM Studioの「Models」タブでロードしたモデル名を確認してください。

- **Q. 会話履歴はどこに保存されますか？**  
  A. `data/` ディレクトリ配下にJSON形式で保存されます（`.gitignore`済み）。

---

## ライセンス

MIT License

---

## 謝辞

- [LM Studio](https://lmstudio.ai/)
- [Streamlit](https://streamlit.io/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

ご意見・ご要望はIssueまたはPRでお知らせください。
