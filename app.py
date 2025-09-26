from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# --------------------------------------------------------------------------------
# 1. LangChainを使用したLLM処理関数の定義
# --------------------------------------------------------------------------------

def generate_llm_response_with_langchain(input_text: str, selected_style_instruction: str) -> str:
    """
    入力テキストとスタイル指定を引数として受け取り、LangChainを使ってLLMからの回答を返す関数。

    Args:
        input_text: ユーザーの入力テキスト。
        selected_style_instruction: ラジオボタンで選択された値（回答の形式指定）。

    Returns:
        LLMの回答テキスト。
    """
    
    # 1. PromptTemplateの定義
    template = """
    あなたは親切なAIアシスタントです。
    以下の「回答スタイル」に従って、ユーザーから提供された「入力テキスト」を処理し、回答を生成してください。

    # 回答スタイル
    {style_instruction}

    # 入力テキスト
    {user_input}

    ---
    上記指示に基づいた回答：
    """
    
    prompt = PromptTemplate(
        input_variables=["style_instruction", "user_input"],
        template=template,
    )

    # 2. LLMの初期化
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 3. チェーンの作成と実行（新しい方法）
    try:
        # プロンプトとLLMを組み合わせたチェーンを作成
        chain = prompt | llm
        
        # チェーンを実行
        response = chain.invoke(
            {
                "style_instruction": selected_style_instruction,
                "user_input": input_text
            }
        )
        
        # レスポンスからテキストを抽出
        return response.content
        
    except Exception as e:
        return f"LangChain/OpenAI APIからの応答エラーが発生しました。詳細: {e}"

# --------------------------------------------------------------------------------
# 2. StreamlitによるUIの定義と関数の利用
# --------------------------------------------------------------------------------

st.title("💡 LangChainデモ：入力フォームとスタイル指定")
st.markdown("LangChainを使って、入力テキストとラジオボタンの指定をプロンプトに組み込み、LLMの回答を取得します。")

# ラジオボタンでの選択値（スタイル指定）の定義
style_options = {
    "箇条書きで要約": "回答を最大3つの**箇条書き**で要約してください。",
    "専門家風に説明": "回答を**アカデミックな専門家**のように、詳細かつ厳密な言葉遣いで説明してください。",
    "小学生に説明": "回答を**小学3年生**にも理解できるように、**やさしい言葉と例**を使って説明してください。",
    "俳句にする": "内容を表現した**俳句（五七五）**を一句作成してください。"
}

st.header("1. 回答スタイルを選択してください")
selected_key = st.radio("選択可能なスタイル:", list(style_options.keys()))
selected_style_instruction = style_options[selected_key] # プロンプトに渡すための具体的な指示文

# 入力フォーム（テキストエリア）
st.header("2. 処理したいテキストを入力してください")
input_text = st.text_area(
    "入力フォーム:",
    "人工知能（AI）の急速な発展は、私たちの生活、仕事、社会構造に大きな変革をもたらしています。特に、自然言語処理の分野におけるLLMの進化は、情報検索やコンテンツ生成の方法を一変させました。",
    height=150
)

# 実行ボタン
if st.button("🚀 LLMに回答を生成させる (LangChain実行)"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("環境変数 `OPENAI_API_KEY` が設定されていません。実行前に設定してください。")
    elif not input_text:
        st.warning("入力フォームにテキストを入力してください。")
    else:
        with st.spinner("AIが回答を生成中です... (LangChain利用)"):
            # 定義した関数の利用
            llm_answer = generate_llm_response_with_langchain(input_text, selected_style_instruction)
            
        st.header("3. LLMからの回答結果")
        st.info(f"**実行されたスタイル:** {selected_key}")
        st.success(llm_answer)