import requests
import json


class ChatBot(object):
    def __init__(self):
        self.system = {"role": "system", "content": "You are a helpful assistant."}
        self.context = [self.system]
        self.model = "gpt-4"
        with open('/home/yue/桌面/gpt_config.txt') as f:
            self.token = f.readline().strip()
            self.url = f.readline().strip()
        self.model_costs = {
                "gpt-3.5-turbo": {"context": 0.0015, "generated": 0.002},
                "gpt-3.5-turbo-16k": {"context": 0.003, "generated": 0.004},
                "gpt-4": {"context": 0.03, "generated": 0.06},
                "gpt-4-32k": {"context": 0.06, "generated": 0.12},
        }

    def chatCompletion(self, model, messages):
        headers = {'Content-Type': 'application/json', 
                   'x-custom-key': self.token}
        data = json.dumps({"messages":messages, "model":model})

        r = requests.request(
            "POST", self.url,
            headers = headers,
            data = data
        )
        return r
    
    def cost_price(self, model, usage):
        context_tokens = usage["prompt_tokens"]
        generated_tokens = usage["completion_tokens"]

        if "gpt-3.5-turbo-16k" in model:
            model = "gpt-3.5-turbo-16k"
        elif "gpt-3.5-turbo" in model:
            model = "gpt-3.5-turbo"
        elif "gpt-4-32k" in model:
            model = "gpt-4-32k"
        else:
            model = "gpt-4"
        print(model)
        cost = (context_tokens / 1000 * self.model_costs[model]["context"]) + (
        generated_tokens / 1000 * self.model_costs[model]["generated"])
    
        return cost
        
    def ask(self, prompt):
        try:
            self.context.append(
                {"role": "user", "content": prompt},
            )
            response = self.chatCompletion(
                model=self.model, messages=self.context
            )
            answer = json.loads(response.text)['choices'][0]['message']
            price = self.cost_price(json.loads(response.text)['model'], json.loads(response.text)['usage'])
            self.context.append(answer)
            return answer["content"], price
        except Exception as e:
            raise

        return None

    def reset(self):
        self.context = [self.system]


