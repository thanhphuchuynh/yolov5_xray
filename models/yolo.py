import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path
# from matplotlib.pyplot import bar_label
import numpy as np
import matplotlib    
import matplotlib.pyplot as plt
# matplotlib.use('TKAgg')
import torch
import torch.nn as nn

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3, Concat, NMS, autoShape
from models.experimental import MixConv2d, CrossConv
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        
            # import numpy as np
        # matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        # print(type(x),x)
        import cv2
        from torchvision import transforms
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
       
        # fig, axis = plt.subplots(1,2,figsize=(15,5))
        for i in range(self.nl):
            # output.cpu().data.numpy().argmax()
           

            # print(self.m[i],x[i].shape)
          
            x[i] = self.m[i](x[i])  # conv

            # print(x[i][0, :, :, :].data.shape)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # print("ha",x[i][0][1:4].shape)
            # images = x[i][0]
            # tensor_to_pil = transforms.ToPILImage()(x[i][0].squeeze_(0))
            # print(tensor_to_pil.size)
            # img = images.permute(1, 2, 0).cpu().detach().numpy()
            # A = x[i][0].permute(3, 0, 1, 2)[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
            # print(A, A.shape, A[:, :, ::-1].shape)
            # print(x[i][0].permute(3, 0, 1, 2).shape)
            plt.figure(figsize=(20, 10))
            v = 0
            for ii in range(len(x[i][0].permute(3, 0, 1, 2))):
                # plt.subplot(3,7,ii+1)
                # b =  np.array(x[i][0].permute(3, 0, 1, 2)[ii, :, :, :].permute(1, 2, 0).cpu().detach().numpy()[:, :, ::-1])
                a = x[i][0].permute(3, 0, 1, 2)[ii, :, :, :]
                # .permute(1, 2, 0)
                for u in range(len(a.data)):
                    print(u,a[u, :, :].shape)
                    b =  np.array(1024*a[u, :, :].cpu().detach().numpy())
                    v = ii +  u + 1
                    plt.subplot(6,9,ii +  u + 1)
                    plt.imshow(b.astype('uint8'), cmap='gray')
            # plt.show()
                # print(a.data)
                # plt.imshow(cv2.cvtColor(b.astype('uint8'), cv2.COLOR_BGR2RGB))
                # plt.imshow(b.astype('uint8'))

                # plt.axis("off")
            # print("save/x"+str(i)+"x.png")
            # plt.savefig("/save/x"+str(i)+"x.png")   
            # plt.show()
            # import cv2
  

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

               
                # b =  np.float32(y[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, ::-1])
                # print(y[0].permute(3, 0, 1, 2).cpu().detach().numpy()[:, :, ::-1].shape)


                # from torchvision import transforms
                # from PIL import Image

                # # print((y[0].permute(3, 0, 1, 2)[0, :, :, :].shape))
                
                # print(len(y[0].permute(3, 0, 1, 2)))
                # plt.figure(figsize=(40, 10))
                # for ii in range(len(y[0].permute(3, 0, 1, 2))):
                    # print(ii,y[0].permute(3, 0, 1, 2)[ii, :, :, :],y[0].permute(3, 0, 1, 2)[ii, :, :, :].shape)
                    # plt.subplot(3,7,ii+1)
                    # plt.subplot()
                    # b =  np.array(y[0].permute(3, 0, 1, 2)[ii, :, :, :].permute(1, 2, 0).cpu().detach().numpy()[:, :, ::-1])
                    
                    # print(b,b.shape)
                    # plt.imshow(out.astype('uint8'))
                    # plt.imshow(cv2.cvtColor(b.astype('uint8'), cv2.COLOR_BGR2RGB))
                    # plt.imshow(b.astype('uint8'), cmap='gray')
                    
                    # plt.axis("off")

                # plt.figure(figsize=(50, 10))
                for ii in range(len(x[i][0].permute(3, 0, 1, 2))):
                    # plt.subplot(3,7,ii+1)
                    # b =  np.array(x[i][0].permute(3, 0, 1, 2)[ii, :, :, :].permute(1, 2, 0).cpu().detach().numpy()[:, :, ::-1])
                    a = y[0].permute(3, 0, 1, 2)[ii, :, :, :]
                    # .permute(1, 2, 0)
                    for u in range(len(a.data)):
                        # print(u,a[u, :, :].shape)
                        b =  np.array(1024*a[u, :, :].cpu().detach().numpy())
                        plt.subplot(6, 9, v + u + ii +1)
                        plt.imshow(b.astype('uint8'), cmap='gray')
                
                # plt.show()

                # plt.show()
                
                # plt.savefig("/save/y"+str(i)+"y.png")  
                # plt.show()
            
                # print(y[0])
                z.append(y.view(bs, -1, self.no))
            # plt.show()
        # print(z)

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        # print(self)
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        # print(x)
        if augment:
            # print(x.shape)
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                # print(si,fi)
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                
                
               
               
                # a = np.asarray(xi[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, ::-1], dtype=np.float16)
               
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].cpu().data.numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                # cv2.imshow("image",255 * xi[0].cpu().data.numpy().transpose((1, 2, 0))[:, :, ::-1])
                # cv2.waitKey(0)

                # print(xi.shape)

                # import cv2
                # import matplotlib
                import numpy as np
                # matplotlib.use('TkAgg')
                # import matplotlib.pyplot as plt
                # plt.axis("off")
                # b =  np.float32(xi[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, ::-1])
                # print(b,b.shape,b.dtype,type(b))
                # plt.imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
                # plt.show()
                
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            # print(type(m))
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        # print(self)

        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
