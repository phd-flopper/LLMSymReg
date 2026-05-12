import torch
from models.base import BaseLLM
import os
from openai import OpenAI
from time import time
import json

class LLMPlanner(BaseLLM):
    def __init__(self, config, user_prompt, responses_dir):
        '''
        LLM-Planner
        '''
        super(LLMPlanner, self).__init__(config)
        self.config = config
        self.model = config['LLM']['planner_model']
        with open(config['prompts']['planner_sys_prompt'], 'r') as f:
            self.sys_prompt = f.read()
            f.close()
        self.user_prompt = user_prompt
        self.destination = responses_dir
        os.makedirs(responses_dir, exist_ok=True)
    
    def parse_response(self):
        response = ''
        constraints = ''
        template = ''
        marker = False
        content = self.get_response().choices[0].message.content
        with open(self.destination + f'/{time()}.txt', "w+") as f:
            f.write(content)
            f.seek(0)
            for line in f.readlines():
                if marker:
                    response += line
                if line.find('{') != -1:
                    response += line
                    marker = True
                if line.find('}') != -1:
                    marker = False
            f.seek(0)
            marker = False
            for line in f.readlines():
                if marker:
                    constraints += line
                if 'CONSTRAINTS' in line.upper():
                    marker = True
                elif marker and "#" in line:
                    marker = False
            f.seek(0)
            line = f.readlines()[-1]
            line = line.replace("'", "")
            line = line.replace("**", "^")
            template = [x for x in line.strip("`#[]= ").split(",")]
            f.close()

        out = json.loads(response)
        
    
        return out, constraints, template


class LLMMutator(BaseLLM):
    def __init__(self, config, constraints, responses_dir):
        '''
        LLM-Mutator
        '''
        super(LLMMutator, self).__init__(config)
        self.config = config
        self.model = config['LLM']['mutator_model']
        with open(config['prompts']['mutator_sys_prompt'], 'r') as f:
            self.sys_prompt = f.read()
            f.close()
        self.constraints = constraints
        with open(self.config['prompts']['mutator_user_prompt'], 'r') as f:
            self.user_prompt = f.read().format(
                constraints=self.constraints, ideas=None, operators=None, template=None,
                eqs=[], vars_=['x0'],
                )
            f.close()
        self.destination = responses_dir
        os.makedirs(responses_dir, exist_ok=True)

    def update_user_prompt(self, eqs, vars_, ideas, operators, template):
        with open(self.config['prompts']['mutator_user_prompt'], 'r') as f:
            self.user_prompt = f.read().format(
                constraints=self.constraints, ideas=ideas, eqs=eqs, vars_=vars_, 
                operators=operators, template=template
                )
            f.close()

    def parse_response(self, eqs, vars_, ideas, operators, template):
        self.update_user_prompt(eqs, vars_, ideas, operators, template)
        content = self.get_response().choices[0].message.content
        with open(self.destination + f'/{time()}.txt', "w+") as f:
            f.write(content)
            f.seek(0)
            line = f.readlines()[-1]
            line = line.replace("'", "")
            line = line.replace("**", "^")
            candidates = [x for x in line.strip("[]").split(",")]
            f.close()
        return candidates


class LLMCritic(BaseLLM):
    def __init__(self, config, constraints, responses_dir):
        '''
        LLM-Critic
        '''
        super(LLMCritic, self).__init__(config)
        self.config = config
        self.model = config['LLM']['critic_model']
        with open(config['prompts']['critic_sys_prompt'], 'r') as f:
            self.sys_prompt = f.read()
            f.close()
        self.constraints = constraints
        with open(self.config['prompts']['critic_user_prompt'], 'r') as f:
            self.user_prompt = f.read().format(
                constraints=self.constraints, eqs=[], params=None, telemetry=None,
                vars_=1, template=None, basic_params=None,
                )
            f.close()
        self.destination = responses_dir
        os.makedirs(responses_dir, exist_ok=True)

    def update_user_prompt(self, eqs, params, telemetry, vars_, template, basic_params):
        with open(self.config['prompts']['critic_user_prompt'], 'r') as f:
            self.user_prompt = f.read().format(
                constraints=self.constraints, eqs=eqs, params=params, telemetry=telemetry,
                vars_=vars_, template=template, basic_params=basic_params,
                )
            f.close()

    def parse_response(self, eqs, params, telemetry, vars_, template, basic_params):
        self.update_user_prompt(eqs, params, telemetry, vars_, template, basic_params)
        content = self.get_response().choices[0].message.content
        response = ''
        ideas = ''
        marker = False
        with open(self.destination + f'/{time()}.txt', "w+") as f:
            f.write(content)
            f.seek(0)
            for line in f.readlines():
                if marker:
                    response += line
                if line.find('{') != -1:
                    response += line
                    marker = True
                if line.find('}') != -1:
                    marker = False
            f.seek(0)
            marker = False
            for line in f.readlines():
                if marker:
                    ideas += line
                if 'CONSTRAINTS' in line.upper():
                    marker = True
                elif marker and "#" in line:
                    marker = False
            f.close()
        try:
            out = json.loads(response)
        except:
            out = params
        return out, ideas