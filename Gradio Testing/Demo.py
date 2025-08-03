import gradio as gr
import pandas as pd
import io
import chardet
import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
# from all_agents_tutorials.Completed.Import_LLM import llm
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
# --- Utility functions ---



def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def read_csv(file_obj):
    if file_obj is None:
        return None, "No file uploaded. Please select a CSV file."
    try:
        encoding = detect_encoding(file_obj.name)
        # gradio's file_obj is a file-like object with .name attribute
        df = pd.read_csv(file_obj.name, encoding=encoding)
        # For stats and preview
        stats = {
            "filename": file_obj.name.split("/")[-1],
            "description": "User uploaded file.",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "colnames": list(df.columns)
        }
        return df, stats, ""
    except Exception as e:
        return None, {}, f"Failed to read file: {e}"

def basic_eda(df):
    if df is None:
        return "No data to analyze."
    n_rows, n_cols = df.shape
    columns = df.columns.tolist()
    eda_md = f"""
### Basic EDA Results

**Rows:** {n_rows}  
**Columns:** {n_cols}  
**Column Names:** {', '.join(columns)}  

**Sample Data:**  
{df.head(5).to_markdown(index=False)}

<details>
<summary>Summary Statistics</summary>

{df.describe(include='all').to_markdown()}
</details>
"""
    return eda_md


sample_questions = [
    "How many Projects are in this document?",
    "How many unique values in the last column?",
    "can you provide me the name of the project with maximum number of employees tagged to it?",
    "Provide a summary showing total time logged by each employee in 2025. Rows should be employees, columns should be months.",
    "Who are the top 5 employees with the most time logged in 2025?",
]

# --------------- Q&A AGENT CREATION AND QA FUNCTION ------------

load_dotenv()

openai_api_type = os.getenv("OPENAI_API_TYPE")
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai_api_version = os.getenv("OPENAI_API_VERSION")
Conf_api_key = os.getenv("CONFLUENCES_API_KEY")

# llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

llm = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_version=openai_api_version,
    azure_endpoint=azure_endpoint,
    openai_api_type=openai_api_type,
    temperature=0.2,
    model="gpt-4o-mini",
    verbose=False,
    max_tokens=1000,
)


def make_agent(df):
    """Create an LLM-pandas agent for a specific DataFrame."""
   
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        return_intermediate_steps=True,
    )
    return agent

# The QA function. It takes question and full DataFrame as input.

def ask_agent(question, df):
    if df is None:
        return "No data loaded for Q&A.", "No data loaded for Q&A."
    try:
        agent = make_agent(df)
        result = agent.invoke(question)
        # Result is a dict; see its keys: 'output', 'intermediate_steps', etc.
        reasoning = ""
        # steps = summarize_reasoning(result.get('intermediate_steps', []))
        steps = result.get('intermediate_steps', [])
        if steps:
            if isinstance(steps, list):
                reasoning = "\n".join(str(s) for s in steps)
            else:
                reasoning = str(steps)
        else:
            reasoning = "*No explicit reasoning steps returned.*"
        final_answer = result.get('output', result if isinstance(result, str) else str(result))
        return reasoning, final_answer
    except Exception as e:
        return "Agent error occurred.", f"Error: {e}"

import re

def summarize_reasoning(intermediate_steps):
    """
    Extracts user-friendly, step-by-step reasoning from agent intermediate_steps,
    which may contain AgentActionMessageLog objects with Python code and comments.
    """
    import re

    # If no steps, show fallback
    if not intermediate_steps:
        return "The agent reasoned step by step to answer your question."

    steps = []

    # Try to be robust for both lists of message logs or raw logs
    for step in intermediate_steps:
        # If the step is an AgentActionMessageLog or similar (from langchain output)
        # Try to access the .tool_input['query'] if possible
        query_code = None

        # Try attribute or dict access
        if hasattr(step, "tool_input") and isinstance(step.tool_input, dict):
            query_code = step.tool_input.get('query')
        elif isinstance(step, dict) and "tool_input" in step and isinstance(step["tool_input"], dict):
            query_code = step["tool_input"].get("query")

        # If code found, extract lines starting with '#'
        if query_code:
            for line in query_code.splitlines():
                l = line.strip()
                if l.startswith("#"):
                    comment = l.lstrip("# ").capitalize()
                    if comment and comment not in steps:
                        steps.append(f"- {comment}")
        # As a fallback, try to extract code comments from a stringified step
        elif isinstance(step, str):
            for line in step.splitlines():
                l = line.strip()
                if l.startswith("#"):
                    comment = l.lstrip("# ").capitalize()
                    if comment and comment not in steps:
                        steps.append(f"- {comment}")

    # If some steps were extracted, return as markdown
    if steps:
        return "\n".join(steps)
    else:
        return "The agent reasoned step by step to answer your question."


# --- Gradio App ---

with gr.Blocks() as demo:
    gr.Markdown("# ADA: AI-based Data Analyzer")
    gr.Markdown("Upload a CSV to begin. ADA will help you analyze your data with AI ✨")
    
    # STATE
    df_state = gr.State()
    stats_state = gr.State()
    
    # SCREEN 1: Upload File
    with gr.Row():
        file_in = gr.File(label="Upload your CSV", file_types=[".csv"])
        submit_btn = gr.Button("Submit")
    upload_status = gr.Markdown("")  # for error/success
    
    # SCREEN 2: EDA
    process_btn = gr.Button("Process the data and do Basic EDA", visible=False)
    eda_md = gr.Markdown(visible=False)
    move_to_qa_btn = gr.Button("If satisfied with the Data, lets move to Q&A", visible=False)
    
    # SCREEN 3: Q&A
    with gr.Column(visible=False) as qa_section:
        gr.Markdown("## Here comes LLM based Q&A ✨")
        
        with gr.Row():
            question_box = gr.Textbox(label="Ask your question on the data", lines=2, scale=10)
            answer_btn = gr.Button("Find answers", scale=2)
        
        with gr.Row():
            with gr.Column(scale=1):
                reasoning_md = gr.Markdown("Reasoning steps will appear here.", elem_id="reasoning-md")
            with gr.Column(scale=3):
                answer_md = gr.Markdown("Final answer will appear here.", elem_id="answer-md")
        
        with gr.Row():
            # Sample question buttons
            q_btns = []
            for i, q in enumerate(sample_questions):
                b = gr.Button(q)
                q_btns.append(b)
        
        with gr.Row():
            file_info = gr.Markdown("")
    
    # --- LOGIC ---
    # State: 0 = initial, 1 = upload done, 2 = EDA done, 3 = Q&A

    def handle_upload(file_obj):
        df, stats, msg = read_csv(file_obj)
        if df is None:
            # error
            return None, None, gr.update(value=msg), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            # success: show EDA button
            return df, stats, gr.update(value="File uploaded! Now, process the data for EDA."), \
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    submit_btn.click(
        handle_upload,
        inputs=file_in,
        outputs=[df_state, stats_state, upload_status, process_btn, eda_md, move_to_qa_btn]
    )

    def handle_eda(df):
        if df is None:
            return gr.update(value="No data to analyze."), gr.update(visible=False)
        eda_result = basic_eda(df)
        return gr.update(value=eda_result, visible=True), gr.update(visible=True)
    process_btn.click(
        handle_eda,
        inputs=df_state,
        outputs=[eda_md, move_to_qa_btn]
    )
    
    def show_qa_section(stats):
        info_text = ""
        if stats:
            info_text = f"""**File:** {stats['filename']}  
**Rows:** {stats['rows']}  
**Columns:** {stats['columns']}  
**Column Names:** {', '.join(stats['colnames'])}"""
        return gr.update(visible=True), gr.update(value=info_text)
    move_to_qa_btn.click(
        show_qa_section,
        inputs=stats_state,
        outputs=[qa_section, file_info]
    )
    
    # --- Sample question buttons autofill textbox ---
    for i, btn in enumerate(q_btns):
        btn.click(lambda q=sample_questions[i]: gr.update(value=q),
                  outputs=question_box)

    # --- Dummy Q&A function (replace with your agent logic later) ---
    def dummy_answer_fn(question, df):
        # A simple mock for now: just echo question and say 'Not implemented'
        reason = f"Received question: **{question}**\n\n_(Reasoning steps would be shown here)_"
        answer = "**Q&A agent not implemented yet.**"
        return gr.update(value=reason), gr.update(value=answer)
            
    answer_btn.click(
        ask_agent,
        inputs=[question_box, df_state],
        outputs=[reasoning_md, answer_md]
    )

demo.launch(server_name="127.0.0.1", server_port=7000, share=False)
