import torch

class DualKenga:
    def __init__(self, model_mml, model_sml):
        self.model_mml = model_mml
        self.model_sml = model_sml
    
    def infer(self, input_ids):
        self.model_mml.eval()
        self.model_sml.eval()
        with torch.no_grad():
            mml_output = self.model_mml(input_ids)
            sml_output = self.model_sml(input_ids)
        return {"MML": mml_output, "SML": sml_output} 