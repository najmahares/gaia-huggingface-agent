import os
import re
import gradio as gr
import requests
import pandas as pd
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, Tool, LiteLLMModel

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class WikipediaSearchTool(Tool):
    name = "wikipedia_search"
    description = "Search Wikipedia for a given query to get factual information."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query string"
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        import wikipedia
        try:
            return wikipedia.summary(query, sentences=5)
        except Exception as e:
            return f"Wikipedia search failed: {e}"

GAIA_SYSTEM_PROMPT = """You are an expert AI agent solving the GAIA benchmark.
Your answers are graded by EXACT STRING MATCH. Follow these rules strictly:
1. Return ONLY the final answer. No explanations, no preamble, no "The answer is".
2. For numbers, use digits (e.g., "42" not "forty-two").
3. For names, use exact spelling and capitalization.
4. For yes/no questions, return exactly "Yes" or "No".
5. If a list is requested, return comma-separated values.
6. Use the web search and wikipedia tools to find facts before answering.
"""

class AdvancedAgent:
    def __init__(self):
        print("Initializing AdvancedAgent...")
        
        self.model = LiteLLMModel(
            model_id="groq/openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )
        
        self.tools = [
            DuckDuckGoSearchTool(max_results=5, rate_limit=2.0),
            WikipediaSearchTool(),
        ]
        
        self.agent = ToolCallingAgent(
            model=self.model,
            tools=self.tools,
            max_steps=10,
            verbosity_level=1,
        )
        
        self.agent.prompt_templates["system_prompt"] = GAIA_SYSTEM_PROMPT + "\n\n" + self.agent.prompt_templates["system_prompt"]
        print("Agent initialized successfully.")

    def __call__(self, question: str) -> str:
        print(f"Agent reasoning on: {question[:80]}...")
        try:
            answer = self.agent.run(question)
            answer = str(answer).strip()
            
            preamble_patterns = [
                r"(?i)^the answer is[:\s]+",
                r"(?i)^final answer[:\s]+",
                r"(?i)^based on .*?,\s*",
            ]
            for pattern in preamble_patterns:
                answer = re.sub(pattern, "", answer).strip()
                
            return answer
        except Exception as e:
            print(f"Agent error: {e}")
            return "Unable to determine"

def run_and_submit_all(username: str, agent_code: str):
    if not username or not agent_code:
        return "Please enter both your Username and Agent Code URL.", None

    print(f"Running for user: {username}")
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = AdvancedAgent()
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
        except Exception as e:
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

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
    gr.Markdown("# GAIA Advanced Agent Runner")
    
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