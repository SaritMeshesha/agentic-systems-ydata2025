# LLM-powered Data Analyst Agent

This Streamlit application uses an LLM-powered agent to analyze the Bitext Customer Support LLM Chatbot Training Dataset. The agent can answer user questions about the dataset, performing both structured (quantitative) and unstructured (qualitative) analysis.

## Features

- Ask questions about the customer support dataset
- Support for different types of analysis:
  - Structured (Quantitative): Category frequencies, examples, intent distributions
  - Unstructured (Qualitative): Summarize categories, analyze intents
- Scope detection to identify if questions are answerable from the dataset
- Support for follow-up questions
- Toggle between planning modes:
  - Pre-planning + Execution: First classify the question, then execute the response
  - ReActive Dynamic Planning: Let the LLM dynamically plan and execute the response

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
4. Enter your OpenAI API key when prompted

## Example Questions

- "What are the most frequent categories?"
- "Show examples of billing category"
- "What categories exist in the dataset?"
- "Summarize the technical support category"
- "What are the common intents in the billing category?"
- "How do agents typically respond to refund requests?"

## Requirements

- Python 3.8+
- OpenAI API key (gpt-4o model access)
- Internet connection (to download the dataset) 