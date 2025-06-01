import os
import random


random.seed(1958)
class Prompt_summary:
    def __init__(self, prompt_path) -> None:
        assert os.path.isfile(prompt_path), "Please specify a prompt template"
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        self.templates = [p.strip() for p in raw_prompts]
        self.response = ''

    def __str__(self) -> str:
        prompt = self.templates[random.randint(0, len(self.templates)-1)]
        prompt = prompt.replace("[explanation]", self.response)     
        return prompt
