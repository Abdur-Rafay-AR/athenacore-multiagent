from .base import call_llm, log_agent_response

def run(topic: str, query: str):
    prompt = f"Research question: {query}\nRespond factually and concisely."
    answer = call_llm(prompt)
    log_agent_response(topic, "Research Agent", answer)
    return answer
