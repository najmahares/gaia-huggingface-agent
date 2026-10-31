import os
import re
import time
import gradio as gr
import requests
import pandas as pd
from ddgs import DDGS
from groq import Groq

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class FastAgent:
    def __init__(self):
        print("Initializing FastAgent...")
        # Initialize the Groq client directly
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def __call__(self, question: str) -> str:
        print(f"Searching for: {question[:60]}...")
        
        # 1. Perform the web search
        search_context = ""
        try:
            # Get top 3 results from DuckDuckGo
            results = list(DDGS().text(question, max_results=3))
            for r in results:
                search_context += f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}\n\n"
        except Exception as e:
            search_context = "No search results found."

        # 2. Ask the super-fast Groq model to read the context and answer
        prompt = f"""You are solving the GAIA benchmark. Your answer is graded by EXACT STRING MATCH.
        
Web Search Results:
{search_context}

Question: {question}

Rules for your answer:
- Return ONLY the exact answer. No explanations, no conversational text.
- If it's a number, use digits (e.g., 42).
- If it's a name, use exact spelling.
- If it's a list, use comma-separated values.
- If the search results do not contain the answer, use your best guess.

Answer:"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant", # Extremely fast model on Groq
                temperature=0.0,
                max_tokens=50
            )
            answer = response.choices[0].message.content.strip()
            
            # Clean up any accidental formatting
            answer = re.sub(r"(?i)^the answer is[:\s]+", "", answer).strip()
            answer = re.sub(r"(?i)^final answer[:\s]+", "", answer).strip()
            return answer
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            return "Unable to determine"

def run_and_submit_all(username: str, agent_code: str):
    if not username or not agent_code:
        return "Please enter both your Username and Agent Code URL.", None

    print(f"Running for user: {username}")
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = FastAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        return f"Error fetching questions: {e}", None

    results_log = []
    answers_payload = []
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
            print(f"Answer: {submitted_answer}")
        except Exception as e:
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})
        
        # Small pause to prevent DDG from blocking us
        time.sleep(2)

    if not answers_payload:
        return "Agent did not produce any answers.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code.strip(), "answers": answers_payload}
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}%\n"
            f"Message: {result_data.get('message', '')}"
        )
        return final_status, pd.DataFrame(results_log)
    except requests.exceptions.RequestException as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)

with gr.Blocks() as demo:
    gr.Markdown("# GAIA Fast Agent Runner")
    
    username_input = gr.Textbox(label="Hugging Face Username", placeholder="e.g., najmahares")
    agent_code_input = gr.Textbox(label="Agent Code URL (GitHub)", placeholder="e.g., https://github.com/your-username/your-repo")

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        inputs=[username_input, agent_code_input],
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("Launching Gradio Interface...")
    demo.launch(debug=True, share=False)