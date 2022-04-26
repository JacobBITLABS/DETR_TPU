"""
Module to run before the main.py for training\n
Defined in train2.sh
"""
import torch, torchvision

def load_model():
    print("Loading Model")
    pretrained = True

    if pretrained:
        # Get pretrained weights
        checkpoint = torch.hub.load_state_dict_from_url(
                    url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                    map_location='cpu',
                    check_hash=True)

        # Remove class weights
        del checkpoint["model"]["class_embed.weight"]
        del checkpoint["model"]["class_embed.bias"]

        # SaveOGH
        torch.save(checkpoint,
                'detr-r50_no-class-head.pth')


def prep_train():
    print("Prepping Training Environment")
    load_model()

    print("Done.")

prep_train()