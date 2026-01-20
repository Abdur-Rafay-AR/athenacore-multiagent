from agents import (
    research_agent,
    summarizer_agent,
    devil_agent,
    insight_agent
)

def run_agent(agent: str, topic: str, query: str = ""):
    if agent == "Research":
        return research_agent.run(topic, query)
    elif agent == "Summarizer":
        return summarizer_agent.run(topic)
    elif agent == "Devil":
        return devil_agent.run(topic)
    elif agent == "Insight":
        return insight_agent.run(topic)
    return "Unknown agent."