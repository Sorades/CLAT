from enum import Enum
from typing import Union

from . import cait, vit
from .cait import CaiTConcept
from .vit import ViTConcept


class EncoderType(Enum):
    CAIT_XS24_384_CONCEPT = "cait_xs24_384_concept"
    CAIT_XXS24_384_CONCEPT = "cait_xxs24_384_concept"
    CAIT_XXS24_224_CONCEPT = "cait_xxs24_224_concept"
    CAIT_S24_224_CONCEPT = "cait_s24_224_concept"
    MIL_VT_CONCEPT = "MIL_VT_Concept"
    VIT_SMALL_CONCEPT = "vit_small_concept"
    VIT_BASE_CONCEPT = "vit_base_concept"

    @property
    def load_func(self):
        return {
            EncoderType.CAIT_XS24_384_CONCEPT: cait.cait_xs24_384_concept,
            EncoderType.CAIT_XXS24_384_CONCEPT: cait.cait_xxs24_384_concept,
            EncoderType.CAIT_XXS24_224_CONCEPT: cait.cait_xxs24_224_concept,
            EncoderType.CAIT_S24_224_CONCEPT: cait.cait_s24_224_concept,
            EncoderType.MIL_VT_CONCEPT: vit.MIL_VT_Concept,
            EncoderType.VIT_SMALL_CONCEPT: vit.vit_small_concept,
            EncoderType.VIT_BASE_CONCEPT: vit.vit_base_concept,
        }[self]


def load_encoder(
    name: str,
    num_classes: int,
    num_lesions: int,
    img_size: int,
    pretrained: bool = True,
) -> Union[CaiTConcept, ViTConcept]:
    model = EncoderType(name).load_func(
        num_classes=num_classes,
        num_lesions=num_lesions,
        img_size=img_size,
        pretrained=pretrained,
    )
    return model
