import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio

class StupidModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )
        self.decoder = nn.Linear(8 * 16, num_classes)
    
    def forward(self, batch):
        return self.decoder(self.encoder(batch))


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

    
class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN(nn.Module):

    def __init__(self, C, num_classes=9):

        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        
        self.fc7 = nn.Linear(192, num_classes)


    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        y = self.fc7(x)

        return x, y
    
    
# class ECAPA_TDNN_Attention(nn.Module):
#     def __init__(self, C, num_classes=9):
#         super(ECAPA_TDNN, self).__init__()

#         self.torchfbank = torch.nn.Sequential(
#             PreEmphasis(),            
#             torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
#                                                  f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
#             )

#         self.specaug = FbankAug() # Spec augmentation

#         self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
#         self.relu   = nn.ReLU()
#         self.bn1    = nn.BatchNorm1d(C)
#         self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
#         self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
#         self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
#         # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
#         self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
# #         self.attention = nn.Sequential(
# #             nn.Conv1d(4608, 256, kernel_size=1),
# #             nn.ReLU(),
# #             nn.BatchNorm1d(256),
# #             nn.Tanh(), # I add this layer
# #             nn.Conv1d(256, 1536, kernel_size=1),
# #             nn.Softmax(dim=2),
# #             )
# #         self.bn5 = nn.BatchNorm1d(3072)
# #         self.fc6 = nn.Linear(3072, 192)
# #         self.bn6 = nn.BatchNorm1d(192)
#         self.bn5 = nn.BatchNorm(1536)
#         self.fc6 = nn.Linear(1536, 192)
#         self.bn7 = nn.BatchNorm(192)
#         self.cls = torch.nn.Parameter(torch.rand(192))
#         self.mha8 = nn.MultiheadAttention(embed_dim=192, num_heads=2, dropout=0.2, batch_first=True)
        
#         self.fc9 = nn.Linear(192, num_classes)


#     def forward(self, x, aug=True):
#         with torch.no_grad():
#             x = self.torchfbank(x)+1e-6
#             x = x.log()   
#             x = x - torch.mean(x, dim=-1, keepdim=True)
#             if aug == True:
#                 x = self.specaug(x)

#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.bn1(x)

#         x1 = self.layer1(x)
#         x2 = self.layer2(x+x1)
#         x3 = self.layer3(x+x1+x2)

#         x = self.layer4(torch.cat((x1,x2,x3),dim=1))
#         x = self.relu(x)

# #         t = x.size()[-1]

# #         global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
# #         w = self.attention(global_x)

# #         mu = torch.sum(x * w, dim=2)
# #         sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

# #         x = torch.cat((mu,sg),1)
# #         x = self.bn5(x)
# #         x = self.fc6(x)
# #         x = self.bn6(x)
# #         y = self.fc7(x)
        
#         x = torch.transpose(x, 1, 2)
#         x = self.fc6(x)
#         x = torch.cat((cls, x), dim=0)
#         x = self.mha8(x,x,x)
#         y = x[:, 0]
        
#         return x, y
    

#VIT    
MAX_LENGTH = 5000

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    
class AccentTransformer(nn.Module):
    def __init__(self, num_classes, emb_size, hidden_size, n_layers=1, n_head=4, dropout=0.1):
        super().__init__()

        self.positional = PositionalEncoding(d_model=emb_size, dropout=dropout, max_len=MAX_LENGTH)
        self.cls = torch.nn.Parameter(torch.rand(1, emb_size))
        
        self.projector = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_size),
            nn.LayerNorm(emb_size),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(emb_size, dim_feedforward=hidden_size, nhead=n_head, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
            
        self.register_buffer("position_ids", torch.arange(MAX_LENGTH).unsqueeze(1))
        
        
    def forward(self, input):
        output = input.squeeze(1).transpose(1,2)
        output = self.projector(output)
        output = torch.cat((self.cls.repeat(output.size(0), 1, 1), output), dim=1)
        output = self.positional(output)
        emb = self.encoder(src=output)[:, 0]
        output = self.decoder(emb)
        return emb, output
    

#QUARTZNET    
class TSC(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, n_groups=1, 
                 dilation=1):
        super(TSC, self).__init__()
        self.tsc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, 
                      dilation=dilation, groups=in_channels,
                      padding=dilation * (kernel_size  - 1) // 2 ),
            nn.Conv1d(in_channels, out_channels, 1, groups=n_groups),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = self.tsc(x)
        return x  


class TSCActivated(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, n_groups=1, 
                 dilation=1):
        super(TSCActivated, self).__init__()
        self.tsc = TSC(kernel_size, in_channels, out_channels, n_groups, 
                       dilation)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.tsc(x)
        x = self.activation(x)
        return x  


class TSCBlock(nn.Module):
    def __init__(self, n_blocks, kernel_size, in_channels, out_channels,
                 n_groups=1, is_intermediate=False):
        super(TSCBlock, self).__init__()
        if is_intermediate:
            in_channels = out_channels
        self.n_blocks = n_blocks
        self.tsc_list = nn.ModuleList([TSCActivated(kernel_size, in_channels, out_channels, n_groups)])
        self.tsc_list.extend([TSCActivated(kernel_size, out_channels, out_channels, n_groups) 
                                  for i in range(1, self.n_blocks-1)])
        self.tsc_list.append(TSC(kernel_size, out_channels, out_channels, n_groups))
        self.pnt_wise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=n_groups)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x_res = self.bn(self.pnt_wise_conv(x))
        for layer in self.tsc_list:
            x = layer(x)
        return self.relu(x + x_res)


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dilation=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                      padding=dilation * (kernel_size - 1) // 2, dilation=dilation, 
                      stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class QuartzNet(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.config = config
        self.net = nn.Sequential(
            TSCActivated(*config['c1']),
            TSCBlock(*config['b1']),
            TSCBlock(*config['b2']),
            TSCBlock(*config['b3']),
            TSCBlock(*config['b4']),
            TSCBlock(*config['b5']),
            TSCActivated(*config['c2']),
            TSCActivated(*config['c3']),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ClassificationNet(nn.Module):
    def __init__(self, num_classes=9, hidden_dim=1024, attn_dim=512, n_mels=80):
        super().__init__() 
        config = {
            #  k, in, out, dilation
            'c1': [33, n_mels, 256, 1],
            'c2': [87, 512, 512, 2],
            'c3': [1, 512, 1024, 1],
            # n_blocks, k, in, out
            'b1': [5, 33, 256, 256],
            'b2': [5, 39, 256, 256],
            'b3': [5, 51, 256, 512],
            'b4': [5, 63, 512, 512],
            'b5': [5, 75, 512, 512]
        }
        self.encoder = QuartzNet(config)
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.LayerNorm(attn_dim),
        )
        self.positional = PositionalEncoding(d_model=attn_dim, dropout=0.1, max_len=MAX_LENGTH)
        self.cls = torch.nn.Parameter(torch.rand(1, 1, attn_dim))
        self.attention = torch.nn.MultiheadAttention(attn_dim, 1, batch_first=True)
        self.out = nn.Linear(attn_dim, num_classes)
        
        
    def forward(self, x):
        output = x.squeeze(1)
        output = self.encoder(output)
        output = output.transpose(1, 2)
        output = self.projector(output)
        output = torch.cat((self.cls.repeat(output.size(0), 1, 1), output), dim=1)
        output, attn_weights = self.attention(output, output, output)
        emb = output[:, 0]
        output = self.out(emb)
        return emb, output #, attn_weights[:, 0]
