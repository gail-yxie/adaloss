from timm.models.vision_transformer import VisionTransformer
from timm import create_model


def vits16r224(num_classes=10) -> VisionTransformer:
    return create_model("vit_small_patch16_224", num_classes=num_classes,
                        pretrained=True, img_size=224)


def swsl_resnet50(num_classes=10):
    return create_model('swsl_resnet50', num_classes=num_classes,
                        pretrained=True)
