import os
import random


random.seed(1958)
class Prompt_explain:
    def __init__(self, prompt_path) -> None:
        assert os.path.isfile(prompt_path), "Please specify a prompt template"
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        self.templates = [p.strip() for p in raw_prompts]
            
        self.historyList = []
        self.trueSelection = ""

    def __str__(self) -> str:
        prompt = self.templates[random.randint(0, len(self.templates)-1)]
        history = ", ".join(self.historyList)
        prompt = prompt.replace("[USER HISTORY]", history)
        prompt = prompt.replace("[ITEM]", self.trueSelection)

            
        return prompt
