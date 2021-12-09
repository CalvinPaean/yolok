import torch
import torch.nn as nn
 
class TestModule(nn.Module):
    def __init__(self):
        super(TestModule,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(16,32,3,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32,10)
        )
 
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
 
model = TestModule()
 
# for name, module in model.named_children():
#     print('children module:', name)
 
for k, v in model.named_modules():
    if isinstance(v, nn.BatchNorm2d):
        print(f'BN: {k}---> {v}\n')
    else:
        print(f'{k}---> {v}\n')