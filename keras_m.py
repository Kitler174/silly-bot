import os
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch_directml as dml

device = dml.device(0)

# === Dataset do tokenizacji on-the-fly
class FileTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': enc['input_ids'].squeeze(0)
        }

INFO_PATH = "info.txt"

def load_progress():
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, "r") as f:
            try:
                return int(f.read().strip())
            except:
                return 0
    return 0

def save_progress(step):
    with open(INFO_PATH, "w") as f:
        f.write(str(step))

# === Funkcja do ładowania tekstu z pliku
def load_texts_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    texts = []
    try:
        if ext == ".txt":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "__label__" in line:
                        line = line.split("__label__")[0].strip()
                    texts.append(line)

        elif ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            for col in df.columns:
                if df[col].dtype == object:
                    for val in df[col].dropna().astype(str):
                        if "__label__" in val:
                            val = val.split("__label__")[0].strip()
                        texts.append(val)

    except Exception as e:
        print(f"Błąd przy {path}: {e}")
    return texts


# === Główna pętla treningowa
def train_on_file(path, tokenizer, model_path=None, output_dir="./checkpoints", save_name="checkpoint"):
    print(f"Trenowanie na pliku: {path}")
    texts = load_texts_from_file(path)
    if not texts:
        print(f"Brak danych w {path}, pomijam.")
        return
    else:
        print("ilość próbek: ", len(texts))

    dataset = FileTextDataset(texts, tokenizer,max_len=64)
    model = GPT2LMHeadModel.from_pretrained(model_path or "distilgpt2").to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # tylko 1 epoka na plik
        per_device_train_batch_size=32,
        fp16=True,
        gradient_accumulation_steps=2,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/{save_name}")
    tokenizer.save_pretrained(f"{output_dir}/{save_name}")
    print()
    return f"{output_dir}/{save_name}"  # ścieżka do nowego checkpointa

if __name__ == "__main__":
    base_path = "polish-data"
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model_checkpoint = None
    all_files = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith((".txt", ".csv", ".tsv")) and "test" not in file.lower():
                all_files.append(os.path.join(root, file))

    # === Wczytaj numer etapu z info.txt
    current_step = load_progress()
    print(f"Rozpoczynamy od etapu {current_step}")

    # === Jeśli mamy progres, ustaw ostatni checkpoint
    if current_step > 0:
        last_checkpoint_tag = f"stage_{current_step - 1:02d}"
        model_checkpoint = os.path.join("./checkpoints", last_checkpoint_tag)
        print(f"Wczytano ostatni checkpoint: {model_checkpoint}")

    for i in range(current_step, len(all_files)):
        file_path = all_files[i]
        print(f"\n[Etap {i+1}/{len(all_files)}] Trening na pliku: {file_path}")
        checkpoint_tag = f"stage_{i:02d}"
        
        # Trening na jednym pliku
        model_checkpoint = train_on_file(
            file_path,
            tokenizer,
            model_path=model_checkpoint,
            save_name=checkpoint_tag
        )

        # Zapisz aktualny etap do info.txt
        save_progress(i + 1)

    # === Zapisz finalny model
    if model_checkpoint:
        final_model_path = "./final_model"
        print(f"\n💾 Zapisuję końcowy model do {final_model_path}")
        final_model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
        final_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

    print("\n✅ Zakończono wszystkie etapy!")
