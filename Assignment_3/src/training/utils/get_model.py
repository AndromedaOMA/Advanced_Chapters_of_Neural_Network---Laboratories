import timm
import sys


def get_model(name):
    if name == 'ResNet50':  # https://huggingface.co/timm/resnet50.a1_in1k
        model = timm.create_model('resnet50.a1_in1k', pretrained=True)
        print('Model loaded!')
        return model
    elif name == 'ResNet18':
        pass
    elif name == 'MLP':
        pass
    else:
        print('The network name you have entered is not supported!')
        sys.exit()
