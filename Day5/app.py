import streamlit as st
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# 🔐 Hardcoded Gemini API Key
GEMINI_API_KEY = "AIzaSyBYTDmlXHHg9SGYDBA-mimwffrYuSruLZc"

# 🔍 DuckDuckGo Search Tool
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for answering questions about current events or recent facts"
    )
]

# 🤖 Gemini Chat Model (Gemini Pro, via LangChain wrapper)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# 💬 Memory for context (optional)
memory = ConversationBufferMemory(memory_key="chat_history")

# 🧠 LangChain Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    memory=memory
)

# 🎨 Streamlit UI
st.set_page_config(page_title="Ask Anything 🤔", page_icon="🧠")
st.title("Ask Anything 🤖📡")
st.markdown("Type your question about current events or facts below:")

# 🧾 User Input
user_input = st.text_input("Your Question:", placeholder="e.g. Who won the IPL 2025 final?")

# 🔍 Trigger Button
if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        try:
            with st.spinner("Thinking... 💭"):
                response = agent.run(user_input)
                st.success("Here's the answer:")
                st.write(response)
        except Exception as e:
            st.error("Oops! Something went wrong while processing your request.")
            st.exception(e)
