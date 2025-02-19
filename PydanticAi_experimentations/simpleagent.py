from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel


model = OllamaModel(model_name='mistral', base_url='http://localhost:11434/v1/')
agent = Agent(  
    model,
    system_prompt='Be concise, reply with one sentence.', 
    
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.data)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""