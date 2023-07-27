import torch


class ModelAction(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )
        self.sigma = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

    def forward(self, state):
        state = self.s(state)
        return self.mu(state), self.sigma(state).exp()


#定义SAC算法
class SAC:

    def __init__(self):
        self.model_action = ModelAction()

        self.model_value = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        self.model_value_next = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.model_value_next.load_state_dict(self.model_value.state_dict())

        self.optimizer_action = torch.optim.Adam(
            self.model_action.parameters(), lr=5e-4)
        self.optimizer_value = torch.optim.Adam(self.model_value.parameters(),
                                                lr=5e-3)

        self.model_action.train()
        self.model_value.train()

        self.requires_grad(self.model_value_next, False)

    def soft_update(self, _from, _to):
        for _from, _to in zip(_from.parameters(), _to.parameters()):
            value = _to.data * 0.995 + _from.data * 0.005
            _to.data.copy_(value)

    def get_action_entropy(self, state):
        mu, sigma = self.model_action(torch.FloatTensor(state).reshape(-1, 3))
        dist = torch.distributions.Normal(mu, sigma)

        action = dist.rsample()
        entropy = dist.log_prob(action) - (1 - action.tanh()**2 + 1e-8).log()
        entropy = -entropy

        return action, entropy

    def requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad_(value)

    def get_loss_cql(self, state, next_state, value):
        return 0

    def train_value(self, state, action, reward, next_state, over):
        self.requires_grad(self.model_value, True)
        self.requires_grad(self.model_action, False)

        #计算target
        with torch.no_grad():
            #计算动作和熵
            next_action, entropy = self.get_action_entropy(next_state)

            #评估next_state的价值
            input = torch.cat([next_state, next_action], dim=1)
            target = self.model_value_next(input)

        #加权熵,熵越大越好
        target = target + 5e-3 * entropy
        target = target * 0.99 * (1 - over) + reward

        #计算value
        value = self.model_value(torch.cat([state, action], dim=1))

        loss = torch.nn.functional.mse_loss(value, target)
        loss += self.get_loss_cql(state, next_state, value)

        loss.backward()
        self.optimizer_value.step()
        self.optimizer_value.zero_grad()
        self.soft_update(self.model_value, self.model_value_next)

        return loss.item()

    def train_action(self, state):
        self.requires_grad(self.model_value, False)
        self.requires_grad(self.model_action, True)

        #计算action和熵
        action, entropy = self.get_action_entropy(state)

        #计算value
        value = self.model_value(torch.cat([state, action], dim=1))

        #加权熵,熵越大越好
        loss = -(value + 5e-3 * entropy).mean()

        #使用model_value计算model_action的loss
        loss.backward()
        self.optimizer_action.step()
        self.optimizer_action.zero_grad()

        return loss.item()