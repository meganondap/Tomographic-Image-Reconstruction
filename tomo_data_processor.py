from neuralop.data.transforms.data_processors import DataProcessor


class ImageDataProcessor(DataProcessor):
    """
    Minimal DataProcessor for image-to-image FNO training.

    Responsibilities:
    - Move input and target tensors to the correct device (CPU/GPU)
    - Call the model on the input
    - Optionally post-process outputs (we keep it simple here)
    """

    def __init__(self, device="cpu"):
        """
        Parameters
        ----------
        device : str
            "cpu" or "cuda" depending on where the model and data should live.
        """
        super().__init__()
        self.device = device
        self.model = None  # will be set by wrap(model)

    def preprocess(self, sample):
        """
        Move tensors in the sample dict to the chosen device.

        Expected keys:
        - sample["x"]: input tensor [B, C, H, W]
        - sample["z"]: target tensor [B, C, H, W]
        """
        x = sample["x"].to(self.device)
        z = sample["y"].to(self.device)
        sample["x"] = x
        sample["y"] = z
        return sample

    def postprocess(self, out, sample):
        """
        Optionally transform outputs and/or targets after model forward.

        Here we do nothing (no inverse normalization), just return as-is.
        """
        return out, sample

    def to(self, device):
        """
        Update the device used by this DataProcessor.
        """
        self.device = device
        return self

    def wrap(self, model):
        """
        Attach the model to this DataProcessor.

        The Trainer will call data_processor(model_input) and this class
        will handle both preprocessing and model forward.
        """
        self.model = model

    def forward(self, sample):
        """
        Complete forward pass through the data processor and model.

        Steps:
        1. Preprocess sample (move to device)
        2. Call model on sample["x"]
        3. Postprocess outputs (no-op here)
        """
        sample = self.preprocess(sample)   # move x, z to device
        x = sample["x"]                    # input tensor
        out = self.model(x)                # model prediction
        out, sample = self.postprocess(out, sample)
        return out, sample