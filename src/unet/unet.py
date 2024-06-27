import torch

class Encoder(torch.nn.Module):
    '''
    Downscale input
    '''
    def __init__(self,in_c,out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.c = torch.nn.Conv2d(in_c,out_c,kernel_size = (3,3),
                                 stride = 1, padding = 0, dilation = 1)
        self.p = torch.nn.MaxPool2d((3,3), stride=None, padding=0, dilation=1)

    def forward(self,x):
        x_cat = self.c(x)
        x_cat = torch.nn.functional.relu(x_cat)
        x = self.p(x_cat)
        return x, x_cat

class Decoder(torch.nn.Module):
    '''
    Upscale input with skip connections
    '''
    def __init__(self,in_c,out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.u = torch.nn.ConvTranspose2d(
                                        in_channels=32,
                                        out_channels=32//2,
                                        #kernel_size=3,
                                        kernel_size=2,
                                        #stride=3,
                                        stride=2,
                                        padding=0,
                                        output_padding=0,
                                        dilation=1
                                    )
        self.c = torch.nn.Conv2d(in_c,out_c,kernel_size = (3,3),
                                 stride = 1, padding = 0, dilation = 1)

    def forward(self,x,skip):
        x_u = self.u(x)
        x_u = torch.nn.functional.interpolate(skip,(skip.shape[-2],skip.shape[-1]))

        x_u = torch.concat([x_u,skip],dim=1)
        print(x_u.shape)
        #self.c = torch.nn.Conv2d(x_u.shape[1],self.out_c,kernel_size = (3,3),
        #                     stride = 1, padding = 0, dilation = 1)

        x = self.c(x_u)
        return x
