import gradio as gr
import httpx
import json
import os
import traceback

API_BASE_URL = "http://localhost:8000"

def extract_text(message):
    """Deeply extract text from Gradio's potential structured message format."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return message.get("text", str(message))
    if isinstance(message, list):
        if len(message) > 0 and isinstance(message[0], dict):
            return message[0].get("text", str(message))
        return str(message)
    return str(message)

def format_job(job):
    title = job.get("title", "Untitled")
    company = job.get("company_name", "Unknown Company")
    location = job.get("location")
    is_remote = job.get("remote", False)
    
    location_str = location if location else ("Remote" if is_remote else "Location not specified")
    if is_remote and location and "remote" not in location.lower():
        location_str += " (Remote)"
        
    apply_url = job.get("apply_url", "#")
    score = job.get("score", 0)
    
    return f"### {title}\n**Company:** {company} | **Location:** {location_str} | **Match Score:** {score:.2f}\n[Apply Here]({apply_url})\n---"

async def process_chat(message, history, session_id):
    # Ensure message is a string
    message = extract_text(message)
    print(f"DEBUG: Processing message: '{message}' | Session: '{session_id}'")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if not session_id:
                print(f"DEBUG: Initial search for query: {message}")
                response = await client.post(
                    f"{API_BASE_URL}/search",
                    json={"query": message, "top_k": 10}
                )
            else:
                print(f"DEBUG: Refinement for session {session_id} with query: {message}")
                response = await client.post(
                    f"{API_BASE_URL}/refine/{session_id}",
                    json={"query": message}
                )
            
            print(f"DEBUG: Status code: {response.status_code}")
            if response.status_code != 200:
                print(f"DEBUG: Error response: {response.text}")
                return f"Error from backend: {response.text}", session_id
            
            data = response.json()
            new_session_id = data.get("session_id", session_id)
            results = data.get("results", [])
            print(f"DEBUG: Found {len(results)} results")
            
            if not results:
                analysis = data.get("query_analysis", {})
                applied = analysis.get("applied_filters", {})
                filters_info = f" with filters {applied}" if applied else ""
                return f"I couldn't find any jobs matching your criteria{filters_info} in the current dataset. Try a broader search like 'Software Engineer'!", new_session_id
            
            output = "I found these jobs for you:\n\n"
            for job in results:
                output += format_job(job)
                
            analysis = data.get("query_analysis", {})
            if analysis.get("applied_filters"):
                filters_str = ", ".join([f"{k}: {v}" for k, v in analysis["applied_filters"].items()])
                output += f"\n\n*Applied Filters: {filters_str}*"
                
            return output, new_session_id
    except Exception as e:
        print(f"DEBUG: Exception in process_chat: {str(e)}")
        print(traceback.format_exc())
        return f"App Error: {str(e)}", session_id

async def bot(history, session_id):
    # history is a list of dicts: [{"role": "user", "content": "..."}, ...]
    user_message = history[-1]["content"]
    # Pass conversation context as list of strings for backend history
    context = [extract_text(h["content"]) for h in history[:-1]] 
    bot_message, new_session_id = await process_chat(user_message, context, session_id)
    history.append({"role": "assistant", "content": bot_message})
    return history, new_session_id

with gr.Blocks() as demo:
    gr.Markdown("# ☕ Cafe Search API - Conversational Job Search")
    
    session_id_state = gr.State("")
    
    chatbot = gr.Chatbot(label="Chat with Cafe Search")
    msg = gr.Textbox(placeholder="What kind of job are you looking for?")
    clear = gr.Button("Clear Session")

    def user(user_message, history):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, session_id_state], [chatbot, session_id_state]
    )
    
    def reset_session():
        return [], ""
    
    clear.click(reset_session, None, [chatbot, session_id_state], queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
