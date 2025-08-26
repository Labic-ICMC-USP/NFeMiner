import unsloth
import pandas as pd, os, json
from unsloth import FastModel
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from .enrichment import NFeMinerBaseGenerateModel

class ProgressCallback(TrainerCallback):
    """
    Custom callback to report training progress.

    This callback calculates the training progress as a float between 0.0 and 1.0
    and passes it to a user-defined function.

    Args:
        progress_bar_callback (Callable[[float], None]): A function to receive progress updates.
    """

    def __init__(self, progress_bar_callback):
        self.progress_bar_callback = progress_bar_callback

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each training step.

        Args:
            args: Training arguments.
            state: Current state of the Trainer.
            control: Trainer control flow object.
            **kwargs: Additional unused keyword arguments.
        """
        if self.progress_bar_callback and state.max_steps:
            progress = state.global_step / state.max_steps
            self.progress_bar_callback(progress)

class NFeFinetuner:
    """
    Handles fine-tuning of a model on NF-e data using configuration parameters.

    The configuration dictionary should include the following keys:
      - "FastModel.from_pretrained": Parameters to load the base model.
      - "FastModel.get_peft_model": Parameters to configure PEFT (Parameter-Efficient Fine-Tuning).
      - "SFTConfig": Training configuration parameters.
      - "dataset_path": Path to the dataset (optional, not used here).
      - "model_output_path": Path to save the trained model.

    Args:
        df (pd.DataFrame): DataFrame with 'prompt' and 'json' columns.
        config (dict): Dictionary containing model and training configurations.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Initializes the NFeFinetuner by preprocessing the dataset.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'prompt' and 'json' columns.
            config (dict): Configuration dictionary for model and training.
        """
        self.config = config
        dataset = Dataset.from_pandas(df)
        self.dataset = dataset.map(
            lambda examples: {
                "text": [
                    f"{NFeMinerBaseGenerateModel.instruction}\n{inp}\n{out}{self.tokenizer.eos_token}"
                    for inp, out in zip(examples["prompt"], examples["json"])
                ]
            },
            batched=True
        )

    def train(self, progress_callback=None):
        """
        Trains the model using the provided dataset and configuration.

        Args:
            progress_callback (Callable[[float], None], optional): Function to receive progress updates. Defaults to None.

        Returns:
            dict: Training statistics returned by the Trainer.
        """
        callbacks = []
        if progress_callback:
            callbacks.append(ProgressCallback(progress_callback))

        model, self.tokenizer = FastModel.from_pretrained(**self.config["FastModel.from_pretrained"])
        self.model = FastModel.get_peft_model(model, **self.config["FastModel.get_peft_model"])
        config = SFTConfig(dataset_text_field="text", **self.config["SFTConfig"])

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            eval_dataset=None,
            args=config,
            callbacks=callbacks
        )

        return trainer.train()

    def save(self):
        """
        Saves the model and tokenizer to the configured output path.

        Saves both the regular model and a merged version (e.g., for Hugging Face Hub compatibility).
        """
        output_dir = self.config['model_output_path']
        self.model.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained_merged(f"{output_dir}_hf", self.tokenizer, save_method="merged_16bit")

    def finetune(self, progress_callback=None):
        """
        Runs the full fine-tuning pipeline: training followed by saving.

        Args:
            progress_callback (Callable[[float], None], optional): Function to receive progress updates. Defaults to None.

        Returns:
            dict: Training statistics.
        """
        train_stats = self.train(progress_callback)
        self.save()
        return train_stats

if __name__ == "__main__":
    config_path = "config.finetuning.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
        df = pd.read_json(config['dataset_path'])
    finetuner = NFeFinetuner(df, config)
    stats = finetuner.finetune()
    print("Training completed!", stats)