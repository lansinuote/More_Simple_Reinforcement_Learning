import torch


class DDPG:

    def __init__(self):
        #初始化模型
        self.model_action = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Tanh(),
        )
        
        self.model_action_delay = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Tanh(),
        )

        self.model_value = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        self.model_value_delay = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        self.optimizer_action = torch.optim.Adam(
            self.model_action.parameters(), lr=5e-4)
        self.optimizer_value = torch.optim.Adam(self.model_value.parameters(),
                                                lr=5e-3)

        self.model_action_delay.load_state_dict(self.model_action.state_dict())
        self.model_value_delay.load_state_dict(self.model_value.state_dict())

        self.requires_grad(self.model_action_delay, False)
        self.requires_grad(self.model_value_delay, False)
        self.model_action.train()
        self.model_value.train()

    def soft_update(self):

        def f(_from, _to):
            for _from, _to in zip(_from.parameters(), _to.parameters()):
                value = _to.data * 0.995 + _from.data * 0.005
                _to.data.copy_(value)

        f(self.model_action, self.model_action_delay)
        f(self.model_value, self.model_value_delay)

    def requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad_(value)

    def train_action(self, state):
        self.requires_grad(self.model_action, True)
        self.requires_grad(self.model_value, False)

        #首先把动作计算出来
        action = self.model_action(state)

        #使用value网络评估动作的价值,价值是越高越好
        input = torch.cat([state, action], dim=1)
        loss = -self.model_value(input).mean()

        loss.backward()
        self.optimizer_action.step()
        self.optimizer_action.zero_grad()

        return loss.item()

    def train_value(self, state, action, reward, next_state, over):
        self.requires_grad(self.model_action, False)
        self.requires_grad(self.model_value, True)

        #计算value
        input = torch.cat([state, action], dim=1)
        value = self.model_value(input)

        #计算target
        with torch.no_grad():
            next_action = self.model_action_delay(next_state)
            input = torch.cat([next_state, next_action], dim=1)
            target = self.model_value_delay(input)
        target = target * 0.99 * (1 - over) + reward

        #计算td loss,更新参数
        loss = torch.nn.functional.mse_loss(value, target)

        loss.backward()
        self.optimizer_value.step()
        self.optimizer_value.zero_grad()

        return loss.item()