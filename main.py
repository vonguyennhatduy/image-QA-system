import os
from tempfile import NamedTemporaryFile
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate

from tool import ImageCaptionTool, ObjectDetectionTool


##############################
### initialize agent #########
##############################

tools = [ImageCaptionTool(), ObjectDetectionTool()]

memory = ConversationBufferWindowMemory(
    k=5,
    return_messages=True
)

llm = ChatOllama(model="phi3", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that can analyze images.\n\n"
     "You have access to the following tools:\n"
     "{tools}\n\n"
     "Tool names: {tool_names}\n\n"
     "You MUST follow this exact format (no extra text):\n\n"
     "Thought: <your reasoning>\n"
     "Action: <one tool name from {tool_names}>\n"
     "Action Input: <a single string input to the tool>\n"
     "Observation: <tool result>\n"
     "... (repeat Thought/Action/Action Input/Observation as needed) ...\n"
     "Thought: I now know the final answer\n"
     "Final: <final answer to the user>\n\n"
     "Rules:\n"
     "- Use exactly the keys: Thought, Action, Action Input, Observation, Final\n"
     "- Do not wrap tool outputs in ```\n"
     "- If you can't or don't need to use a tool, go straight to Final.\n"
     ),
    ("human", "{input}\n\n{agent_scratchpad}"),
])

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

################################
######## Streamlit UI ###########
################################

st.title('Ask a question to an image')
st.header("Please upload an image")

file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    st.image(file, use_column_width=True)

    user_question = st.text_input('Ask a question about your image:')

    if user_question:
        with NamedTemporaryFile(dir='.', delete=False, suffix='.jpg') as f:
            f.write(file.getbuffer())
            image_path = f.name
        

        with st.spinner(text="In progress..."):
            try:
                response = agent_executor.invoke({
                    "input": f"{user_question}, this is the image path: {image_path}"
                })
                st.write(response["output"])
            finally:
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        print(f"Warning: Không thể xóa file tạm {image_path}: {e}")