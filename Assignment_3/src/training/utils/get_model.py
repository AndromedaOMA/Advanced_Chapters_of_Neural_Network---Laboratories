import timm
import sys


def get_model(name):
    name = name.lower()
    if name == 'resnet50':  # https://huggingface.co/timm/resnet50.a1_in1k
        model = timm.create_model('resnet50.a1_in1k', pretrained=True)
        # model = timm.create_model("hf_hub:timm/resnet50.a1_in1k", pretrained=True)
        print('Model loaded!')
        return model
    elif name == 'resnet18':
        pass
    elif name == 'resnest14d':
        pass
    elif name == 'resnest26d':
        pass
    elif name == 'MLP':
        pass
    else:
        print('The network name you have entered is not supported!')
        sys.exit()


get_model('ResNet50')
