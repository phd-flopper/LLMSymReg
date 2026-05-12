import torch
from openai import OpenAI

class BaseGenerator(torch.nn.Module):
    def __init__(self):
        '''
        Base class for SR wrappers
        '''
        super(BaseGenerator, self).__init__()
        self.pareto_frontier = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def get_best_equation(self):
        return self.pareto_frontier


class BaseLLM(torch.nn.Module):
    def __init__(self, config):
        '''
        Base class for LLM wrappers
        '''
        super(BaseLLM, self).__init__()
        self.config = config
        self.client = OpenAI(
            base_url=config['env']['url'],
            api_key=config['env']['token'],
        )
        self.model = None
        self.sys_prompt = None
        self.user_prompt = None

    def get_response(self):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": self.user_prompt}
            ],
        )
        return completion

    def parse_response(self):
        raise NotImplementedError