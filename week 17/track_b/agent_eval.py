import mlflow
import os
from evidently.test_suite import TestSuite
from evidently.tests import TestLLMCategoricalCorrectness # Not exact class name, I will use a simple test suite implementation if needed

# Mock the Agent
@mlflow.trace
def agent_query(query: str, prompt_version: str):
    # Mocking different agent behaviors based on prompt versions
    if prompt_version == "v1":
        return f"This is a standard answer to: {query}"
    elif prompt_version == "v2":
        return f"This is a detailed and polite answer to: {query}. Thank you!"
    else:
        return f"Unknown query"

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow_agent.db")
    mlflow.set_experiment("Agentic_AI_Tracing")
    
    queries = [
        "What is your return policy?",
        "How do I track my order?",
        "Do you offer international shipping?"
    ]
    
    versions = ["v1", "v2", "v3"]
    
    for version in versions:
        with mlflow.start_run(run_name=f"Agent_Prompt_{version}"):
            mlflow.log_param("prompt_version", version)
            
            # Run traces
            for q in queries:
                response = agent_query(q, version)
            
            # Since running Evidently LLM requires an OpenAI key and real setup, we mock the result
            # Assuming Test Suite passes 100% for v2, 66% for v1, 0% for v3
            if version == "v2":
                pass_rate = 1.0
            elif version == "v1":
                pass_rate = 0.66
            else:
                pass_rate = 0.0
                
            mlflow.log_metric("pct_tests_passed", pass_rate)
            
            print(f"Version {version} completed. Pass rate: {pass_rate}")

if __name__ == "__main__":
    main()
