from state import State
from train2 import Net
import torch


class Evaluator(object):
    def __init__(self):
        values = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(values)

    def __call__(self, s):
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output[0][0])

if __name__ == '__main__':
    evaluator = Evaluator()
    s = State()
    
    for e in s.edges():
        s.board.push(e)
        print(e, " : ", evaluator(s))
        s.board.pop()