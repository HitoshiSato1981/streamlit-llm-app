# app.py
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ----------------------------
# LLM呼び出し関数（条件の必須要件）
# ----------------------------
def ask_llm(user_text: str, expert_choice: str) -> str:
    """
    入力テキスト(user_text) と ラジオ選択値(expert_choice) を受け取り、
    LLMの回答文字列を返す。
    """
    system_prompts = {
        "健康アドバイザー": (
            "あなたは優秀な健康アドバイザーです。"
            "ユーザーの質問に対して、要点を整理し、実務で使える形で回答してください。"
            "不明点がある場合は、前提を置いて回答し、最後に確認質問を1〜3個添えてください。"
        ),
        "栄養士": (
            "あなたは優秀な栄養士です。"
            "ユーザーの質問に対して、初心者にも分かるように噛み砕いて説明してください。"
            "必要に応じて例えや箇条書きを使ってください。"
        ),
    }
    system_message = system_prompts.get(expert_choice, system_prompts["健康アドバイザー"])

    llm = ChatOpenAI(
        model="gpt-4.1-mini",   # 好みのモデルに変更OK
        temperature=0.2,
    )

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_text),
    ]

    response = llm.invoke(messages)
    return response.content


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="LangChain LLM Demo", page_icon="🤖")
st.title("🤖 健康 × 食事 Webアプリ")

st.markdown(
    """
### このアプリでできること
- **質問入力**すると専門家AIが回答をしてくれます。  
- **ボタン**で「健康アドバイザー/栄養士」を選び、選択に応じて **専門家**を切り替えます。  
- 返ってきた **回答を画面に表示**します

### 操作方法
1. 「専門家タイプ」を **健康アドバイザーまたは栄養士** から選択  
2. 下の入力欄に質問や相談内容を入力  
3. 「送信」を押すと、回答が下に表示されます
"""
)

# 入力フォーム（1つ）
with st.form("query_form"):
    expert_choice = st.radio(
        "専門家タイプを選択",
        options=["健康アドバイザー", "栄養士"],
        horizontal=True,
    )
    user_text = st.text_area("入力フォーム（ここに質問を入力）", height=140, placeholder="例：最近眠れないことが多いのですが、何かアドバイスはありますか？")
    submitted = st.form_submit_button("送信")

if submitted:
    if not user_text.strip():
        st.warning("入力が空です。質問文を入力してください。")
    else:
        with st.spinner("LLMが回答を生成中..."):
            try:
                answer = ask_llm(user_text=user_text, expert_choice=expert_choice)
                st.subheader("回答")
                st.write(answer)
            except Exception as e:
                st.error("エラーが発生しました。APIキー設定や依存関係を確認してください。")
                st.exception(e)

st.divider()
st.caption("© 2024 健康 × 食事 Webアプリ")