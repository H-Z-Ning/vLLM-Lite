import streamlit as st
import requests
import time

# 配置后端 URL
BACKEND_URL = "http://localhost:8001/v1/chat/completions"

st.set_page_config(page_title="VLLM-Lite Chat", page_icon="🤖")
st.title("🚀 VLLM-Lite 推理终端")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("输入您的问题..."):
    # 添加用户消息到界面
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用后端接口
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*正在思考...*")
        
        try:
            payload = {
                "model": "vllm-lite",
                "messages": st.session_state.messages,
                "max_tokens": 512
            }
            
            response = requests.post(BACKEND_URL, json=payload, timeout=130)
            
            if response.status_code == 200:
                full_response = response.json()["choices"][0]["message"]["content"]
                message_placeholder.markdown(full_response)
                # 保存助手回答
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"连接后端失败: {e}")
