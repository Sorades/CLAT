from typing import Dict, List, Tuple
from lightning import LightningModule

        
class AbstractCPTModule(LightningModule):
    def __init__(self, disease_names: List[str], lesion_names: List[str], img_size: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_disease = len(disease_names)
        self.num_lesions = len(lesion_names)
        self.disease_names = disease_names
        self.lesion_names = lesion_names
        self.img_size = img_size
        
    def _run_step(self, batch, **kwargs) -> Tuple[Dict, Dict]:
        raise NotImplementedError(
            f"The method _run_step must be implemented in {self.__class__.__name__}"
        )
    
    def training_step(self, batch):
        loss_dict, rs_dict = self._run_step(batch)
        disease_logits, disease_lbls = rs_dict["disease_logits"], batch["disease_lbls"]
        lesion_logits, lesion_lbls = rs_dict["lesion_logits"], batch["lesion_lbls"]

        for name, value in loss_dict.items():
            self.log(
                f"loss/train_{name}",
                value,
                prog_bar=True if name == "loss" else False,
                on_step=False,
                on_epoch=True,
                batch_size=disease_logits.shape[0],
                sync_dist=True,
            )

        outs = {
            "loss": loss_dict["loss"],
            "disease_logits": disease_logits.detach(),
            "disease_lbls": disease_lbls.detach(),
            "lesion_logits": lesion_logits.detach(),
            "lesion_lbls": lesion_lbls.detach(),
        }
        
        return outs

    def validation_step(self, batch):
        loss_dict, rs_dict = self._run_step(batch)
        disease_logits, disease_lbls = rs_dict["disease_logits"], batch["disease_lbls"]
        lesion_logits, lesion_lbls = rs_dict["lesion_logits"], batch["lesion_lbls"]
        
        for name, value in loss_dict.items():
            self.log(
                f"loss/val_{name}",
                value,
                prog_bar=True if name == "loss" else False,
                on_step=False,
                on_epoch=True,
                batch_size=disease_logits.shape[0],
                sync_dist=True,
            )

        outs = {
            "disease_logits": disease_logits.detach(),
            "disease_lbls": disease_lbls.detach(),
            "lesion_logits": lesion_logits.detach(),
            "lesion_lbls": lesion_lbls.detach(),
        }
        
        return outs

    def test_step(self, batch, **kwargs):
        _, rs_dict = self._run_step(batch, **kwargs)
        disease_logits, disease_lbls = rs_dict["disease_logits"], batch["disease_lbls"]
        lesion_logits, lesion_lbls = rs_dict["lesion_logits"], batch["lesion_lbls"]
        
        outs = {
            "disease_logits": disease_logits.detach(),
            "disease_lbls": disease_lbls.detach(),
            "lesion_logits": lesion_logits.detach(),
            "lesion_lbls": lesion_lbls.detach(),
        }

        return outs
