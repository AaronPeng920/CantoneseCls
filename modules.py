import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
    
    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
            
        return x

class CRMD(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 conv_size=3, conv_stride=1, conv_padding=1, 
                 pool_size=2, pool_stride=2, drop_rate=0.25):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_size, conv_stride, conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, pool_stride),
            nn.Dropout(drop_rate)
        )
        
    def forward(self, x):
        return self.block(x)

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 conv_size=3, conv_stride=2, conv_padding=1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_size, conv_stride, conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)

class MSFF(nn.Module):
    def __init__(self, in_channels, mid_channels1, mid_channels2, out_channels):
        super().__init__()
        
        self.cbr = CBR(3 * mid_channels2, out_channels)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels1, kernel_size=1),
            nn.BatchNorm2d(mid_channels1),
            nn.ReLU(),
            nn.Conv2d(mid_channels1, mid_channels2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels2),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels1, kernel_size=1),
            nn.BatchNorm2d(mid_channels1),
            nn.ReLU(),
            nn.Conv2d(mid_channels1, mid_channels2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels2),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels1, kernel_size=1),
            nn.BatchNorm2d(mid_channels1),
            nn.ReLU(),
            nn.Conv2d(mid_channels1, mid_channels2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels2),
            nn.ReLU()
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        out = torch.concat((b1, b2, b3), dim=1)
        out = self.cbr(out)
        
        return out

class FirstLevelNet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.crmd1 = CRMD(in_channels, mid_channels[0])
        self.crmd2 = CRMD(mid_channels[0], mid_channels[1])
        self.msff = MSFF(mid_channels[1], mid_channels[2], mid_channels[3], mid_channels[4])
        self.crmd3 = CRMD(mid_channels[1], mid_channels[5])
        self.crmd4 = CRMD(mid_channels[4]+mid_channels[5], out_channels)
        
    def forward(self, x):
        out = self.crmd1(x)
        out = self.crmd2(out)
        
        out1 = self.msff(out)
        out2 = self.crmd3(out)
        out = torch.concat((out1, out2), dim=1)
        
        out = self.crmd4(out)
        
        return out

# input torch.Size([32, 1, 8, 161])    
class SecondLevelNet(nn.Module):
    def __init__(self, in_channels, mid_channels,  
                 input_size, lstm_hidden_size1=256, lstm_hidden_size2=128,
                 cls_count=10, latent_features=512, droprate=0.25):
        super().__init__()
        
        self.upper_crmd1 = CRMD(in_channels, mid_channels[0])
        self.upper_crmd2 = CRMD(mid_channels[0], mid_channels[1])
        self.upper_crmd3 = CRMD(mid_channels[1], mid_channels[2])
        self.upper_flatten = Flatten()
        
        self.lower_flatten = Flatten()
        self.lower_lstm1 = nn.LSTM(input_size, hidden_size=lstm_hidden_size1, batch_first=True)
        self.lower_lstm2 = nn.LSTM(input_size=lstm_hidden_size1, hidden_size=lstm_hidden_size2, batch_first=True)
        self.lower_dropout = nn.Dropout(droprate)
        
        self.later_dropout1 = nn.Dropout(droprate * 2)
        # You should change 56
        self.later_dense1 = Dense(56 + lstm_hidden_size2, out_features=latent_features, activation=nn.ReLU())
        self.later_dropout2 = nn.Dropout(droprate)
        self.later_dense2 = Dense(in_features=latent_features, out_features=cls_count, activation=nn.Softmax(dim=-1))
        
        
    def forward(self, x):
        out1 = self.upper_crmd1(x)
        out1 = self.upper_crmd2(out1)
        out1 = self.upper_crmd3(out1)
        out1 = self.upper_flatten(out1)
        
        out2 = self.lower_flatten(x)
        out2, _ = self.lower_lstm1(out2)
        out2, _ = self.lower_lstm2(out2)
        out2 = self.lower_dropout(out2)
        
        out = torch.concat((out1, out2), dim=1)
        out = self.later_dropout1(out)
        out = self.later_dense1(out)
        out = self.later_dropout2(out)
        out = self.later_dense2(out)
        
        return out
               


if __name__ == '__main__':
    data1 = torch.randn(32, 1, 128, 938).cuda()
    model1 = FirstLevelNet(1, [2, 4, 8, 16, 32, 64], 1).cuda()
    out1 = model1(data1)    # [32, 1, 8, 58]
    print(out1.shape)
    data2 = torch.randn(32, 1, 8, 58).cuda()
    model2 = SecondLevelNet(1, [2, 4, 8], 8 * 58).cuda()
    out2 = model2(data2)
    print(out2.shape)
    
    