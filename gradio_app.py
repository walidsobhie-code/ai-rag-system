"""
AI RAG System - Beautiful Gradio Web UI
"""
import gradio as gr
from rag_engine import RAGEngine

rag = RAGEngine()

def chat(message, history):
    if not message.strip():
        return history, ""
    
    result = rag.chat(message)
    response = result.get("answer", "No response")
    sources = result.get("sources", [])
    
    source_text = ""
    if sources:
        source_text = "\n\n**Sources:**\n" + "\n".join([f"- {s.get('source', 'Unknown')}" for s in sources[:3]])
    
    full_response = response + source_text
    history.append((message, full_response))
    return history, ""

with gr.Blocks(title="🤖 AI RAG System") as demo:
    gr.Markdown("""
    # 🤖 AI RAG System
    ### Ask questions about your documents using local AI (Ollama)
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Ask a question", placeholder="What is machine learning?", lines=2)
            with gr.Row():
                submit_btn = gr.Button("💬 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 💡 Example Questions
            - What is AI?
            - Explain neural networks
            - How does Python work?
            """)

    msg.submit(lambda m, h: chat(m, h), [msg, chatbot], [msg, chatbot])
    submit_btn.click(lambda m, h: chat(m, h), [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: (None, ""), outputs=[chatbot, msg])

demo.launch(server_name="0.0.0.0", server_port=7860)
