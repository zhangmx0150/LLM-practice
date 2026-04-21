import streamlit as st

from rag import PrivateKnowledgeBase

def main():
    st.set_page_config(page_title="私有知识库", page_icon="🏢")
    st.title("🏢 私有知识库")

    # 1. 初始化知识库
    if "kb" not in st.session_state:
        # 这里默认指定向量库路径
        st.session_state.kb = PrivateKnowledgeBase(persist_directory="./vector_db")
        if st.session_state.kb.load_vectorstore():
            st.session_state.kb.setup_qa_chain()
            st.toast("✅ 本地知识库加载成功！")

    # 2. 接收用户提问
    question = st.chat_input("请输入您的问题：")

    if question:
        # 显示用户问题
        with st.chat_message("user"):
            st.write(question)

        # 显示 AI 回答
        with st.chat_message("assistant"):
            if not st.session_state.kb.qa_chain:
                st.error("知识库尚未就绪，请先构建或加载。")
            else:
                # 兼容你刚才改过的“流式输出”打字机效果
                message_placeholder = st.empty()
                full_response = ""
                final_sources = []

                for chunk, sources in st.session_state.kb.ask(question):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    final_sources = sources

                message_placeholder.markdown(full_response)

                # 展示参考来源
                if final_sources:
                    with st.expander("查看参考来源"):
                        for i, src in enumerate(final_sources, 1):
                            st.text(f"--- 来源 {i} ---")
                            st.text(src)

if __name__ == "__main__":
    main()