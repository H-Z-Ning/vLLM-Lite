import streamlit as st
import requests
import time

BACKEND_URL = "http://localhost:8000/v1/chat/completions"

st.set_page_config(page_title="VLLM-Lite Chat", page_icon="🤖")
st.title("🚀 VLLM-Lite 推理终端")

# --- 侧边栏配置 ---
st.sidebar.header("推理设置")
temp = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.8, 0.05)
rep_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.05)
max_tokens = st.sidebar.number_input("Max Tokens", 1, 2048, 512)

if st.sidebar.button("清空历史"):
    st.session_state.messages = []
    st.rerun()

# --- 聊天逻辑 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*正在思考...*")
        
        try:
            payload = {
                "model": "vllm-lite",
                "messages": st.session_state.messages,
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": top_p,
                "repetition_penalty": rep_penalty
            }
            
            response = requests.post(BACKEND_URL, json=payload, timeout=130)
            
            if response.status_code == 200:
                full_response = response.json()["choices"][0]["message"]["content"]
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"连接后端失败: {e}")
