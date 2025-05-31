---
title: LLM Data Analyst Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🤖 LLM-powered Data Analyst Agent

An intelligent data analysis assistant that helps you explore and understand customer support datasets using advanced language models.

## 🌟 Features

- **Interactive Data Analysis**: Ask questions in natural language and get intelligent responses
- **Multiple Planning Modes**: Choose between pre-planning and reactive dynamic planning
- **Beautiful UI**: Modern, responsive interface with custom styling
- **Real-time Conversations**: Chat-like interface for seamless interaction
- **Dataset Insights**: Automatic analysis of customer support conversations

## 🚀 How to Use

1. **Ask Questions**: Type your question about the customer support data
2. **Get Insights**: The AI will analyze the data and provide detailed answers
3. **Explore Further**: Follow up with additional questions for deeper analysis

### Example Questions:
- "What are the most common customer issues?"
- "Show me examples of billing problems"
- "What's the distribution of customer intents?"
- "Summarize the main categories of support requests"

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **AI Model**: Nebius API (Qwen/Qwen3-30B-A3B)
- **Data Processing**: Pandas for data manipulation
- **Dataset**: Bitext Customer Support Dataset

## 📊 Dataset

This app analyzes the [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) which contains real customer support conversations with:

- **Categories**: Different types of customer issues
- **Intents**: Specific customer intentions  
- **Customer Messages**: Original customer inquiries
- **Agent Responses**: Support agent replies

## 🔧 Configuration

The app requires a Nebius API key to function. This has been configured as an environment variable for this Space.

## 💡 Tips

- **Be Specific**: More specific questions often yield better insights
- **Explore Different Angles**: Try both quantitative ("how many") and qualitative ("why") questions
- **Use Follow-ups**: Build on previous answers for deeper analysis

## 🎯 Planning Modes

- **Pre-planning**: The agent first classifies your question, then executes analysis
- **Reactive Planning**: The agent dynamically decides how to approach your question

Choose the mode that works best for your analysis style!

---

*Built with ❤️ using Streamlit and powered by advanced language models* 