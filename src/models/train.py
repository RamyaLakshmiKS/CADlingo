"""
Training Pipeline for CADlingo

Fine-tunes CodeT5 model for text-to-CAD code generation.

This module:
1. Loads preprocessed training data
2. Configures CodeT5 model and tokenizer
3. Implements training loop with evaluation
4. Saves model checkpoints
5. Tracks metrics (BLEU, accuracy, loss)

"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
try:
    from datasets import load_metric
except (ImportError, AttributeError):
    # load_metric deprecated, use evaluate instead
    import evaluate
    load_metric = None


class CADDataset(Dataset):
    """PyTorch Dataset for CAD code generation."""
    
    def __init__(self, data_file: str, tokenizer, max_source_length: int = 128, max_target_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSON dataset file
            tokenizer: Tokenizer instance
            max_source_length: Max tokens for input description
            max_target_length: Max tokens for output code
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Tokenize input (description)
        source = entry['description']
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize output (code)
        target = entry['code']
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace padding token id with -100 for loss calculation)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


class CADTrainer:
    """Trainer for CAD code generation model."""
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5-base",
        data_dir: str = None,
        output_dir: str = None,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: Hugging Face model name
            data_dir: Directory containing processed datasets
            output_dir: Directory to save model and results
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        # Setup paths
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "processed"
        
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "results" / "models"
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Metrics - use evaluate library instead of deprecated load_metric
        if load_metric is not None:
            self.bleu_metric = load_metric('sacrebleu')
        else:
            self.bleu_metric = evaluate.load('sacrebleu')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_bleu': [],
            'learning_rates': []
        }
    
    def load_datasets(self, batch_size: int = 8):
        """
        Load training and validation datasets.
        
        Args:
            batch_size: Batch size for dataloaders
        """
        train_file = self.data_dir / "train_dataset.json"
        val_file = self.data_dir / "val_dataset.json"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"Validation data not found: {val_file}")
        
        print("Loading datasets...")
        train_dataset = CADDataset(str(train_file), self.tokenizer)
        val_dataset = CADDataset(str(val_file), self.tokenizer)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self, optimizer, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Calculate loss
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
                
                # Generate predictions for BLEU
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode predictions and references
                decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Decode labels (replace -100 with pad token)
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = self.tokenizer.pad_token_id
                decoded_labels = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                
                predictions.extend(decoded_preds)
                references.extend([[ref] for ref in decoded_labels])  # BLEU expects list of lists
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate BLEU score
        bleu_result = self.bleu_metric.compute(predictions=predictions, references=references)
        bleu_score = bleu_result['score']
        
        return avg_loss, bleu_score, predictions, references
    
    def train(
        self,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        save_every: int = 2
    ):
        """
        Main training loop.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps for scheduler
            save_every: Save checkpoint every N epochs
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        # Load datasets
        self.load_datasets(batch_size)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_bleu = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_epoch(optimizer, scheduler)
            self.history['train_loss'].append(train_loss)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Evaluate
            val_loss, val_bleu, predictions, references = self.evaluate()
            self.history['val_loss'].append(val_loss)
            self.history['val_bleu'].append(val_bleu)
            
            # Print metrics
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val BLEU: {val_bleu:.2f}")
            
            # Save sample predictions
            if epoch % save_every == 0 or epoch == epochs - 1:
                self.save_samples(predictions[:5], references[:5], epoch)
            
            # Save best model
            if val_bleu > best_bleu:
                best_bleu = val_bleu
                self.save_model("best_model")
                print(f"New best model! BLEU: {val_bleu:.2f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}")
        
        # Save final model
        self.save_model("final_model")
        
        # Plot training history
        self.plot_training_history()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best BLEU Score: {best_bleu:.2f}")
        print(f"Models saved to: {self.output_dir}")
        print("="*60)
    
    def save_model(self, name: str):
        """Save model checkpoint."""
        save_path = self.output_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training history
        history_file = save_path / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_samples(self, predictions: List[str], references: List[List[str]], epoch: int):
        """Save sample predictions to file."""
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        sample_file = samples_dir / f"epoch_{epoch}_samples.txt"
        
        with open(sample_file, 'w') as f:
            f.write(f"Epoch {epoch} - Sample Predictions\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                f.write(f"Sample {i+1}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Reference:\n{ref[0]}\n\n")
                f.write(f"Prediction:\n{pred}\n\n")
                f.write("=" * 80 + "\n\n")
    
    def plot_training_history(self):
        """Plot training metrics."""
        plots_dir = self.output_dir.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot 1: Training and Validation Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: BLEU Score
        axes[0, 1].plot(epochs, self.history['val_bleu'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('BLEU Score')
        axes[0, 1].set_title('Validation BLEU Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary Stats
        axes[1, 1].axis('off')
        summary_text = f"""
        Training Summary
        {'='*30}
        
        Total Epochs: {len(epochs)}
        
        Final Train Loss: {self.history['train_loss'][-1]:.4f}
        Final Val Loss: {self.history['val_loss'][-1]:.4f}
        
        Best BLEU Score: {max(self.history['val_bleu']):.2f}
        Final BLEU Score: {self.history['val_bleu'][-1]:.2f}
        
        Device: {self.device}
        Model: CodeT5-base
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = plots_dir / f"training_history_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nTraining plots saved to: {plot_file}")
        
        plt.close()


def main():
    """Main training function."""
    # Create trainer
    trainer = CADTrainer(
        model_name="Salesforce/codet5-base",
    )
    
    # Train model
    trainer.train(
        epochs=10,           
        batch_size=4,        
        learning_rate=5e-5,
        warmup_steps=100,
        save_every=2
    )


if __name__ == "__main__":
    main()
