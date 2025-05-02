from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from lightning import LightningModule

from clat.data import DataItem
from clat.encoder import load_encoder
from clat.utils import CLATOutput


class KnowledgeGuideLoss(nn.Module):
    def __init__(
        self, knowledge_embeds: torch.Tensor, eps=1e-8, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.knowledge_embeds_T = knowledge_embeds.T
        self.eps = eps

    def forward(self, inputs, targets):
        scores = inputs @ self.knowledge_embeds_T.to(inputs.device)
        gt = torch.arange(targets.size(-1), dtype=torch.long, device=targets.device)
        gt = gt.unsqueeze(0).expand(scores.shape[0], scores.shape[1])
        loss = self.loss(scores, gt)
        loss = torch.mean(
            torch.mean(loss * targets, dim=-1)
            / (torch.sum(targets, dim=-1) + self.eps),
            dim=-1,
        )
        return loss


class CLAT(LightningModule):
    def __init__(
        self,
        disease_names: list[str],
        lesion_names: list[str],
        img_size: int = 224,
        arch_name: str = "cait_s24_224_concept",
        pretrained: bool = True,
        disease_loss_weight: float = 1.0,
        lesion_loss_weight: float = 0.6,
        KG_loss_weight: float = 0.4,
        with_EK: bool = False,
        training_int_prob: Optional[float] = None,
        training_int_milestone: int = 0,
        eval_int: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_disease = len(disease_names)
        self.num_lesions = len(lesion_names)
        self.disease_names = disease_names
        self.lesion_names = lesion_names
        self.img_size = img_size
        self.training_int_prob = training_int_prob
        self.training_int_milestone = training_int_milestone
        self.eval_int = eval_int

        self.model = load_encoder(
            arch_name,
            pretrained=pretrained,
            num_classes=self.num_disease,
            num_lesions=self.num_lesions,
            img_size=img_size,
        )
        if "No DR" in disease_names:
            knowledge_embeds_path = "data/FLAIR_DR_with_EK.pt"
        elif "No RAO" in disease_names:
            knowledge_embeds_path = "data/FLAIR_RAO_with_EK.pt"
        if not with_EK:
            knowledge_embeds_path.replace("with_EK", "without_EK")
        knowledge_embeds = torch.load(knowledge_embeds_path)
        self.token2concept = nn.Linear(self.model.embed_dim, knowledge_embeds.shape[1])
        self.patch_size = self.model.patch_embed.patch_size

        self.disease_loss_weight = disease_loss_weight
        self.lesion_loss_weight = lesion_loss_weight
        self.KG_loss_weight = KG_loss_weight

        self.loss_disease = nn.CrossEntropyLoss()
        self.loss_lesion = nn.MultiLabelSoftMarginLoss()
        self.loss_knowledge_guide = KnowledgeGuideLoss(knowledge_embeds)

    def forward(self, x, return_attn=False, **kwargs) -> CLATOutput:
        return self.model(x, return_attn=return_attn, **kwargs)

    def shared_step(
        self, batch: DataItem, stage: Literal["train", "val", "test"]
    ) -> dict[str, Any]:
        images, disease_lbls, lesion_lbls = (
            batch.image,
            batch.disease_lbls,
            batch.lesion_lbls,
        )
        bs = images.shape[0]

        output: CLATOutput
        # forward
        if (
            self.training
            and self.training_int_prob
            and self.training_int_milestone is not None
            and self.current_epoch >= self.training_int_milestone
        ):
            output = self(
                images, int_prob=self.training_int_prob, lesion_lbls=lesion_lbls
            )
        else:
            output = self(images, return_attn=stage == "test")

        # loss
        disease_loss = (
            self.loss_disease(output.disease_logits, disease_lbls)
            if self.disease_loss_weight > 0
            else 0
        )
        lesion_loss = (
            self.loss_lesion(output.lesion_logits, lesion_lbls)
            if self.lesion_loss_weight > 0
            else 0
        )
        KG_loss = (
            self.loss_knowledge_guide(
                self.token2concept(output.lesion_tokens), lesion_lbls
            )
            if self.KG_loss_weight > 0
            else 0
        )

        loss = (
            self.disease_loss_weight * disease_loss
            + self.lesion_loss_weight * lesion_loss
            + self.KG_loss_weight * KG_loss
        )

        self.log(
            f"loss/{stage}_loss", loss, prog_bar=True, batch_size=bs, sync_dist=True
        )
        self.log_dict(
            {
                f"loss/{stage}_disease_loss": disease_loss,
                f"loss/{stage}_lesion_loss": lesion_loss,
                f"loss/{stage}_KG_loss": KG_loss,
            },
            prog_bar=False,
            batch_size=bs,
            sync_dist=True,
        )

        ret_dict = {"loss": loss, **output._asdict()}

        return ret_dict

    def get_explanations(self, image):
        pred = self(image, return_attn=True)
        disease_logits, lesion_logits, _, _, cross_attn_maps = pred
        disease_confidence, disease_type = torch.max(
            torch.softmax(disease_logits, dim=-1), dim=-1
        )
        disease_type = disease_type.item()
        disease_confidence = disease_confidence.item()
        lesion_confidence = (
            torch.sigmoid(lesion_logits).squeeze().detach().cpu().numpy()
        )
        contributions = (
            cross_attn_maps.sum(dim=1)
            .squeeze()
            .softmax(dim=-1)[disease_type]
            .detach()
            .cpu()
            .numpy()
        )

        expl = self._expl(
            disease_type, disease_confidence, lesion_confidence, contributions
        )

        return expl, pred

    def _expl(
        self, disease_type, disease_confidence, lesion_confidence, contributions
    ) -> str:
        expl = [
            f"Disease diagnosis: {self.disease_names[disease_type]}({disease_confidence:.2%})\n"
        ]
        expl.append("Lesions discovery:\n")
        exist_lesions, absent_lesions = [], []
        for i in range(self.num_lesions):
            if lesion_confidence[i] > 0.5:
                exist_lesions.append(f"{self.lesion_names[i]}({contributions[i]:.2%})")
            else:
                absent_lesions.append(f"{self.lesion_names[i]}({contributions[i]:.2%})")
            expl.append(f"{self.lesion_names[i]}({lesion_confidence[i]:.2%})\n")
        expl.append(f"This is {self.disease_names[disease_type]}. The existence of ")
        for e in exist_lesions:
            if e == exist_lesions[0]:
                expl.append(f"{e}, ")
            elif e == exist_lesions[-1]:
                expl.append(f"and {e}")
            else:
                expl.append(f"{e}, ")
        expl.append("  and the absent of ")
        for a in absent_lesions:
            if a == absent_lesions[0]:
                expl.append(f"{a}, ")
            elif a == absent_lesions[-1]:
                expl.append(f"and {a} confirms this diagnosis.")
            else:
                expl.append(f"{a}, ")
        return "".join(expl)

    def intervene(self, image, intervene_lesion_idx, interven_lesion_probs):
        pred = self(
            image,
            return_attn=True,
            intervene_sample_idx=0,
            intervene_cpt_idx=intervene_lesion_idx,
            lesion_lbls=interven_lesion_probs,
        )
        disease_logits, lesion_logits, _, _, cross_attn_maps = pred
        disease_confidence, disease_type = torch.max(
            torch.softmax(disease_logits, dim=-1), dim=-1
        )
        disease_type = disease_type.item()
        disease_confidence = disease_confidence.item()
        lesion_confidence = (
            torch.sigmoid(lesion_logits).squeeze().detach().cpu().numpy()
        )
        contributions = (
            cross_attn_maps.sum(dim=1)
            .squeeze()
            .softmax(dim=-1)[disease_type]
            .detach()
            .cpu()
            .numpy()
        )

        expl = self._expl(
            disease_type, disease_confidence, lesion_confidence, contributions
        )

        return expl, pred

    def training_step(self, batch):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch):
        return self.shared_step(batch, stage="val")

    def test_step(self, batch):
        return self.shared_step(batch, stage="test")
