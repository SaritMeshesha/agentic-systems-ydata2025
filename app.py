import json
import os
from enum import Enum
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Set up page config with custom styling
st.set_page_config(
    page_title="🤖 LLM Data Analyst Agent",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --error-color: #d62728;
        --warning-color: #ff9800;
        --background-color: #0e1117;
        --card-background: #262730;
    }
    
    /* Custom styling for the main container */
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: var(--card-background);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .success-card {
        background: linear-gradient(90deg, rgba(44, 160, 44, 0.1) 0%, rgba(44, 160, 44, 0.05) 100%);
        border-left: 4px solid var(--success-color);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(90deg, rgba(214, 39, 40, 0.1) 0%, rgba(214, 39, 40, 0.05) 100%);
        border-left: 4px solid var(--error-color);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .quick-actions-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #2c3e50;
    }
    
    .quick-actions-card h3 {
        color: var(--primary-color);
        margin-top: 0;
    }
    
    .quick-actions-card ul {
        margin-bottom: 0;
    }
    
    .quick-actions-card li {
        margin-bottom: 0.5rem;
        color: #495057;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1f77b4 0%, #0e4b7a 100%);
    }
    
    /* Metrics styling */
    .metric-container {
        background: var(--card-background);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(90deg, rgba(31, 119, 180, 0.1) 0%, rgba(31, 119, 180, 0.05) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .assistant-message {
        background: linear-gradient(90deg, rgba(255, 127, 14, 0.1) 0%, rgba(255, 127, 14, 0.05) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--secondary-color);
    }
    
    /* Planning mode styling */
    .planning-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .pre-planning {
        background: rgba(31, 119, 180, 0.2);
        color: var(--primary-color);
        border: 1px solid var(--primary-color);
    }
    
    .reactive-planning {
        background: rgba(255, 127, 14, 0.2);
        color: var(--secondary-color);
        border: 1px solid var(--secondary-color);
    }
    
    /* Animation for thinking indicator */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .thinking-indicator {
        animation: pulse 2s infinite;
    }
    
    /* Improved expander styling */
    .streamlit-expanderHeader {
        background: var(--card-background);
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# API configuration
api_key = os.environ.get("NEBIUS_API_KEY")

if not api_key:
    st.markdown(
        """
    <div class="error-card">
        <h3>🔑 API Key Configuration Required</h3>
        <p>Please configure your Nebius API key to get started:</p>
        <ol>
            <li>Open the <code>.env</code> file in your project directory</li>
            <li>Replace <code>your_api_key_here</code> with your actual Nebius API key</li>
            <li>Save the file and restart the application</li>
        </ol>
        <p><strong>Example .env file:</strong></p>
        <pre>NEBIUS_API_KEY=your_actual_api_key_here</pre>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

# Set the API key in environment for consistency
os.environ["OPENAI_API_KEY"] = api_key

# Nebius API settings
NEBIUS_API_URL = "https://api.studio.nebius.com/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-30B-A3B"


# Function to call Nebius API
def call_nebius_api(messages, response_format=None, thinking_mode=False):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {"model": MODEL_NAME, "messages": messages}

    if response_format:
        payload["response_format"] = response_format

    # If in thinking mode, ask the model to show its reasoning
    if thinking_mode:
        # Add instruction to show thinking process
        last_message = messages[-1]
        enhanced_content = (
            f"{last_message['content']}\n\n"
            f"Important: First explain your thinking process step by step, "
            f"then provide your final answer clearly labeled as 'FINAL ANSWER:'"
        )
        messages[-1]["content"] = enhanced_content
        payload["messages"] = messages

    try:
        response = requests.post(NEBIUS_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            st.error(f"Response: {e.response.text}")
        return None


# Load Bitext dataset
@st.cache_data
def load_bitext_dataset():
    try:
        dataset = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
        )
        df = pd.DataFrame(dataset["train"])
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


# Define enums for request types
class AnalysisType(str, Enum):
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    OUT_OF_SCOPE = "out_of_scope"


class ColumnType(str, Enum):
    CATEGORY = "category"
    INTENT = "intent"
    CUSTOMER = "customer"
    AGENT = "agent"


# Define schema for agent requests
class AgentRequest(BaseModel):
    question: str = Field(..., description="The user's question")
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    target_columns: Optional[List[ColumnType]] = Field(
        None, description="Columns to analyze"
    )
    is_follow_up: bool = Field(
        False, description="Whether this is a follow-up question"
    )
    previous_context: Optional[str] = Field(
        None, description="Context from previous question"
    )


# Function to classify the user question
def classify_question(
    question: str, previous_context: Optional[str] = None
) -> AgentRequest:
    """
    Use the LLM to classify the question and determine the analysis type and target columns.
    """
    system_prompt = """
    You are a data analyst assistant that classifies user questions about a customer support dataset.
    The dataset contains customer support conversations with these columns:
    - category: The category of the customer query
    - intent: The specific intent of the customer query
    - customer: The customer's message
    - agent: The agent's response
    
    Classify the question into one of these types:
    - quantitative: Questions about statistics, frequencies, distributions, or examples of categories/intents
    - qualitative: Questions asking for summaries or insights about specific categories/intents
    - out_of_scope: Questions that cannot be answered using the dataset
    
    Also identify which columns are relevant to the question.
    
    Return a JSON object with the following fields:
    {
      "analysis_type": "quantitative" | "qualitative" | "out_of_scope",
      "target_columns": ["category", "intent", "customer", "agent"]
    }
    """

    context_info = f"\nPrevious context: {previous_context}" if previous_context else ""

    user_prompt = f"Classify this question: {question}{context_info}"

    response = call_nebius_api(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    if not response:
        # Fallback if API call fails
        return AgentRequest(
            question=question,
            analysis_type=AnalysisType.OUT_OF_SCOPE,
            target_columns=[],
            is_follow_up=bool(previous_context),
            previous_context=previous_context,
        )

    try:
        content = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        )
        result = json.loads(content)

        # Convert string column names to ColumnType enum values
        target_columns = []
        for col in result.get("target_columns", []):
            try:
                target_columns.append(ColumnType(col))
            except ValueError:
                pass  # Skip invalid column types

        return AgentRequest(
            question=question,
            analysis_type=AnalysisType(result.get("analysis_type", "out_of_scope")),
            target_columns=target_columns,
            is_follow_up=bool(previous_context),
            previous_context=previous_context,
        )
    except (json.JSONDecodeError, ValueError) as e:
        st.warning(f"Error parsing API response: {str(e)}")
        return AgentRequest(
            question=question,
            analysis_type=AnalysisType.OUT_OF_SCOPE,
            target_columns=[],
            is_follow_up=bool(previous_context),
            previous_context=previous_context,
        )


# Function to generate a response to the user's question
def generate_response(df: pd.DataFrame, request: AgentRequest) -> str:
    """
    Generate a response to the user's question based on the request classification.
    """
    # Get thinking mode setting from session state
    show_thinking = st.session_state.get("show_thinking", True)

    if request.analysis_type == AnalysisType.OUT_OF_SCOPE:
        return "I'm sorry, but I can't answer that question based on the available customer support data."

    # Prepare context with dataset information
    data_description = f"Dataset contains {len(df)} customer support conversations."

    if request.analysis_type == AnalysisType.QUANTITATIVE:
        # For quantitative questions, prepare relevant statistics
        stats_context = ""
        if ColumnType.CATEGORY in request.target_columns:
            category_counts = df["category"].value_counts().to_dict()
            stats_context += f"\nCategory distribution: {json.dumps(category_counts)}"

        if ColumnType.INTENT in request.target_columns:
            intent_counts = df["intent"].value_counts().to_dict()
            stats_context += f"\nIntent distribution: {json.dumps(intent_counts)}"

        # If specific examples are requested, include sample data
        if "example" in request.question.lower() or "show" in request.question.lower():
            for col in request.target_columns:
                if col.value in df.columns:
                    # Try to extract a specific value the user might be looking for
                    search_terms = [term.lower() for term in df[col.value].unique()]
                    for term in search_terms:
                        if term in request.question.lower():
                            examples = (
                                df[df[col.value].str.lower() == term]
                                .head(5)
                                .to_dict("records")
                            )
                            stats_context += f"\nExamples of {col.value}='{term}': {json.dumps(examples)}"
                            break
    else:  # QUALITATIVE
        stats_context = ""
        # For qualitative questions, prepare relevant data for summarization
        for col in request.target_columns:
            if col.value in df.columns:
                unique_values = df[col.value].unique().tolist()
                stats_context += (
                    f"\nUnique values for {col.value}: {json.dumps(unique_values)}"
                )

                # If there's a specific category/intent mentioned in the question
                for value in unique_values:
                    if value.lower() in request.question.lower():
                        filtered_data = (
                            df[df[col.value] == value].head(10).to_dict("records")
                        )
                        stats_context += f"\nSample data for {col.value}='{value}': {json.dumps(filtered_data)}"
                        break

    # Generate the response using LLM
    system_prompt = f"""
    You are a data analyst assistant that answers questions about a customer support dataset.
    {data_description}
    
    Use the following context to answer the question:
    {stats_context}
    
    Be concise and data-driven in your response. Mention specific numbers and patterns when appropriate.
    If there isn't enough information to fully answer the question, acknowledge that limitation.
    """

    previous_context = ""
    if request.is_follow_up:
        previous_context = (
            f"\nThis is a follow-up to previous context: {request.previous_context}"
        )

    response = call_nebius_api(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question: {request.question}{previous_context}",
            },
        ],
        thinking_mode=show_thinking,
    )

    if not response:
        return "I'm sorry, I encountered an error while processing your question. Please try again."

    return (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "I couldn't generate a response. Please try again.")
    )


# Function to plan and execute approach based on mode
def process_question(
    df: pd.DataFrame, question: str, mode: str, previous_context: Optional[str] = None
) -> str:
    """
    Process the user question using the specified planning mode.
    """
    # Add thinking indicator to the UI with custom styling
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(
        """
    <div class="thinking-indicator">
        <div class="info-card">
            ⚙️ <strong>Agent is thinking...</strong> Analyzing your question and preparing response.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Get thinking mode setting from session state
    show_thinking = st.session_state.get("show_thinking", True)

    if mode == "pre_planning":
        # Pre-planning: First classify, then execute
        request = classify_question(question, previous_context)
        st.session_state.last_request = request

        # Show classification if thinking is enabled
        if show_thinking:
            thinking_placeholder.markdown(
                f"""
            <div class="info-card">
                ⚙️ <strong>Agent classified this as a 
                <span style="color: var(--primary-color);">{request.analysis_type}</span> question</strong>
                <br>📊 Target columns: {[col.value for col in request.target_columns]}
            </div>
            """,
                unsafe_allow_html=True,
            )

        result = generate_response(df, request)
    else:  # reactive_planning
        # Reactive planning: Let the LLM decide approach dynamically
        system_prompt = """
        You are a data analyst assistant that answers questions about a customer support dataset.
        The dataset contains customer support conversations with categories, intents, customer messages, and agent responses.
        
        Analyze the question and determine how to approach it:
        1. Identify if it's asking for statistics, examples, summaries, or insights
        2. Determine which aspects of the data are relevant
        3. Generate a direct and concise response based on the data
        
        If the question cannot be answered with the customer support dataset, politely explain that it's outside your scope.
        """

        # Prepare dataset information
        data_description = f"Dataset with {len(df)} records. "
        data_description += f"Sample of 5 records: {df.sample(5).to_dict('records')}"
        data_description += f"\nColumns: {df.columns.tolist()}"

        # Include full distributions for categories and intents
        # Check if the question is about distributions or frequencies
        question_lower = question.lower()
        include_distributions = any(
            term in question_lower
            for term in [
                "distribution",
                "frequency",
                "count",
                "how many",
                "most frequent",
                "most common",
                "statistics",
            ]
        )

        # Always include category values
        data_description += f"\nCategory values: {df['category'].unique().tolist()}"

        # Include full distribution data if the question appears to need it
        if include_distributions:
            if "category" in question_lower or "categories" in question_lower:
                category_counts = df["category"].value_counts().to_dict()
                data_description += (
                    f"\nCategory distribution: {json.dumps(category_counts)}"
                )

            if "intent" in question_lower or "intents" in question_lower:
                intent_counts = df["intent"].value_counts().to_dict()
                data_description += (
                    f"\nIntent distribution: {json.dumps(intent_counts)}"
                )
            else:
                # Just provide a sample of intents if not specifically asking about them
                data_description += f"\nIntent values sample: {df['intent'].sample(10).unique().tolist()}"
        else:
            # Just provide a sample of intents
            data_description += (
                f"\nIntent values sample: {df['intent'].sample(10).unique().tolist()}"
            )

        context_info = ""
        if previous_context:
            context_info = f"\nThis is a follow-up to: {previous_context}"

        response = call_nebius_api(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nDataset information: {data_description}{context_info}",
                },
            ],
            thinking_mode=show_thinking,
        )

        if not response:
            thinking_placeholder.empty()
            return "I'm sorry, I encountered an error while processing your question. Please try again."

        result = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "I couldn't generate a response. Please try again.")
        )

    # Clear the thinking indicator
    thinking_placeholder.empty()

    # Process the result to separate thinking from final answer if needed
    if show_thinking and "FINAL ANSWER:" in result:
        parts = result.split("FINAL ANSWER:")
        thinking = parts[0].strip()
        final_answer = parts[1].strip()

        # Display thinking and final answer with clear separation
        with st.expander("🧠 Agent's Thinking Process", expanded=True):
            st.markdown(thinking)

        return final_answer
    else:
        return result


# Main app interface
def main():
    # Custom header
    st.markdown(
        """
    <div class="main-header">
        <h1>🤖 LLM-powered Data Analyst Agent</h1>
        <p>Intelligent Analysis of Bitext Customer Support Dataset</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load dataset
    with st.spinner("🔄 Loading dataset..."):
        df = load_bitext_dataset()

    if df is None:
        st.markdown(
            """
        <div class="error-card">
            <h3>❌ Dataset Loading Failed</h3>
            <p>Failed to load dataset. Please check your internet connection and try again.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # Success message with dataset info
    st.markdown(
        f"""
    <div class="success-card">
        <h3>✅ Dataset Loaded Successfully</h3>
        <p>Loaded <strong>{len(df):,}</strong> customer support records ready for analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Planning mode selection with styling
        st.markdown("### 🧠 Planning Mode")
        planning_mode = st.radio(
            "Select how the agent should approach questions:",
            ["pre_planning", "reactive_planning"],
            format_func=lambda x: (
                "🎯 Pre-planning + Execution"
                if x == "pre_planning"
                else "⚡ Reactive Dynamic Planning"
            ),
            help="Choose between structured pre-analysis or dynamic reactive planning",
        )

        # Display current mode with badge
        mode_class = (
            "pre-planning" if planning_mode == "pre_planning" else "reactive-planning"
        )
        mode_name = (
            "Pre-Planning" if planning_mode == "pre_planning" else "Reactive Planning"
        )
        st.markdown(
            f"""
        <div class="planning-badge {mode_class}">
            {mode_name} Mode Active
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Thinking process toggle
        st.markdown("### 🧠 Agent Behavior")
        if "show_thinking" not in st.session_state:
            st.session_state.show_thinking = True

        show_thinking = st.checkbox(
            "🔍 Show Agent's Thinking Process",
            value=st.session_state.show_thinking,
            help="Display the agent's reasoning and analysis steps",
        )
        st.session_state.show_thinking = show_thinking

        st.markdown("---")

        # Dataset stats in sidebar
        st.markdown("### 📊 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Total Records", f"{len(df):,}")
        with col2:
            st.metric("📂 Categories", len(df["category"].unique()))

        st.metric("🎯 Unique Intents", len(df["intent"].unique()))

    # Main content area
    # Dataset information in an expandable section
    with st.expander("📊 Dataset Information", expanded=False):
        st.markdown("### Dataset Details")

        # Create metrics row
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        with metrics_col1:
            st.metric("Total Records", f"{len(df):,}")
        with metrics_col2:
            st.metric("Columns", len(df.columns))
        with metrics_col3:
            st.metric("Categories", len(df["category"].unique()))
        with metrics_col4:
            st.metric("Intents", len(df["intent"].unique()))

        st.markdown("### Sample Data")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("### Category Distribution")
        st.bar_chart(df["category"].value_counts())

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "last_request" not in st.session_state:
        st.session_state.last_request = None

    # User input section
    st.markdown("## 💬 Ask Your Question")

    # Create a more prominent input area
    user_question = st.text_input(
        "What would you like to know about the customer support data?",
        placeholder="e.g., What are the most common customer issues?",
        key="user_input",
        help="Ask questions about statistics, examples, or insights from the dataset",
    )

    # Submit button with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_clicked = st.button("🚀 Analyze Question", use_container_width=True)

    if submit_clicked and user_question:
        # Add user question to conversation
        st.session_state.conversation.append({"role": "user", "content": user_question})

        # Get previous context if this might be a follow-up
        previous_context = None
        if len(st.session_state.conversation) > 2:
            # Get the previous assistant response
            previous_context = st.session_state.conversation[-3]["content"]

        # Process the question with enhanced loading indicator
        with st.spinner("🤖 Agent is analyzing your question..."):
            response = process_question(
                df, user_question, planning_mode, previous_context
            )

        # Add response to conversation
        st.session_state.conversation.append({"role": "assistant", "content": response})

    # Display conversation with styled messages
    if st.session_state.conversation:
        st.markdown("## 💭 Conversation History")

        for i, message in enumerate(st.session_state.conversation):
            if message["role"] == "user":
                st.markdown(
                    f"""
                <div class="user-message">
                    <strong>👤 You:</strong> {message['content']}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="assistant-message">
                    <strong>🤖 Agent:</strong> {message['content']}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if i < len(st.session_state.conversation) - 1:  # Not the last message
                    st.markdown("---")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation = []
            st.rerun()


if __name__ == "__main__":
    main()
