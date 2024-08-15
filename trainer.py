import torch
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import BertTokenizerFast, BertForSequenceClassification

from dataset import AmznReviewsDataset, AmznReviewsDataLoader
from evaluate import get_accuracy, get_precision, get_recall, get_f1

# distributed setup
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class Trainer:

    def __init__(self, data_dir, device, output="details_Brand", learning_rate=3e-3):

        # first fix device as it decides the distributed setup
        self.device = device
        print(f"Using device: {self.device}")
        self.ddp = False
        if isinstance(self.device, int): # if it's ddp then the input device will be int
            self.ddp = True
        
        # Data stuff first
        self.data_dir = data_dir
        ## Get the datasets
        train_dataset = AmznReviewsDataset(data_dir, split="train", output=output)
        val_dataset = AmznReviewsDataset(data_dir, split="val", output=output)
        test_dataset = AmznReviewsDataset(data_dir, split="test", output=output)
        ## Get the dataloaders
        self.train_dataloader = AmznReviewsDataLoader(
            train_dataset, batch_size=4, 
            shuffle=True if self.ddp else False,
            sampler=DistributedSampler(train_dataset) if self.ddp else None
        )
        self.val_dataloader = AmznReviewsDataLoader(
            val_dataset, batch_size=4, shuffle=False,
            sampler=DistributedSampler(val_dataset) if self.ddp else None
        )
        self.test_dataloader = AmznReviewsDataLoader(
            test_dataset, batch_size=4, shuffle=False
        )

        # Model stuff
        self.tokenizer = BertTokenizerFast.from_pretrained("./bert-tokenizer/")
        self.lm = BertForSequenceClassification.from_pretrained("./bert-model/")
        self.lm.eval()
        self.head = torch.nn.Linear(
            self.lm.config.hidden_size, len(train_dataset.out2idx)
        )
        if self.ddp:
            self.lm = DDP(self.lm, device_ids=[device])
            self.head = DDP(self.head, device_ids=[device])

        # Training stuff
        self.criterion = torch.nn.CrossEntropyLoss()
        # freeze the BERT model, only train the head this works better, but feel free to experiment!
        for param in self.lm.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=learning_rate)

    def train_step(self, batch):

        # Set the model to train mode
        self.head.train()

        # Tokenize the input
        inputs = self.tokenizer(
            batch["input"],
            truncation=True,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
        )
        assert all(inputs["input_ids"][:, 0] == self.tokenizer.cls_token_id), "CLS token not present in input"

        # Send the inputs to the device
        inputs.to(self.device)
        targets = batch["output"].to(self.device)

        # Forward pass
        lm_outputs = self.lm(**inputs, output_hidden_states=True)
        cls_output = lm_outputs.hidden_states[-1][:, 0, :]
        out = self.head(cls_output)

        # Calculate loss
        loss = self.criterion(out, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def eval_step(self, batch):

        # Set the model to eval mode
        self.head.eval()

        # Tokenize the input
        inputs = self.tokenizer(
            batch["input"],
            truncation=True,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
        )
        inputs.to(self.device)
        targets = batch["output"].to(self.device)

        # Forward pass
        lm_outputs = self.lm(**inputs, output_hidden_states=True)
        cls_output = lm_outputs.hidden_states[-1][:, 0, :]
        out = self.head(cls_output)

        loss = self.criterion(out, targets)

        outputs = torch.argmax(out, dim=1)

        return outputs, loss.item()

    def train(self, num_epochs=5):

        best_model = None
        best_f1 = 0.0

        for epoch in range(num_epochs):

            print(f"Epoch {epoch+1}/{num_epochs}")

            # Training loop
            train_loss = 0.0
            for i, batch in tqdm(
                enumerate(self.train_dataloader), total=len(self.train_dataloader)
            ):
                loss = self.train_step(batch)
                
                # get loss from all machines if in distributed setup
                if self.ddp:
                    loss = dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    
                train_loss += loss

                if i % 10000 == 9999:
                    train_loss = 0.0
                    if ((self.ddp and self.device == 0) or not self.ddp):
                        print(f"Iteration {i+1}, Loss: {train_loss/10000:.4f}")

            val_outputs = []
            val_targets = []
            # Validation loop
            for batch in tqdm(self.val_dataloader, total=len(self.val_dataloader)):
                outputs, loss = self.eval_step(batch)
                val_outputs.append(outputs)
                val_targets.append(batch["output"])

            if self.ddp:
                # Prepare tensors for gathering
                gathered_outputs = [torch.zeros_like(val_outputs) for _ in range(dist.get_world_size())]
                gathered_targets = [torch.zeros_like(val_targets) for _ in range(dist.get_world_size())]
                
                # Gather tensors from all processes
                dist.all_gather(gathered_outputs, val_outputs)
                dist.all_gather(gathered_targets, val_targets)
                
                # Concatenate gathered tensors
                val_outputs = torch.cat(gathered_outputs).cpu().detach().numpy().reshape(-1)
                val_targets = torch.cat(gathered_targets).cpu().detach().numpy().reshape(-1)
            
            else:    
                val_outputs = torch.cat(val_outputs).cpu().detach().numpy().reshape(-1)
                val_targets = torch.cat(val_targets).cpu().detach().numpy().reshape(-1)

            accuracy = get_accuracy(val_targets, val_outputs)
            precision = get_precision(val_targets, val_outputs)
            recall = get_recall(val_targets, val_outputs)
            f1 = get_f1(val_targets, val_outputs)

            if f1 > best_f1:
                best_f1 = f1
                if self.ddp:
                    dist.barrier()
                    best_model = [self.lm.module.state_dict(), self.head.module.state_dict()]
                else:
                    best_model = [self.lm.state_dict(), self.head.state_dict()]

            if ((self.ddp and self.device == 0) or not self.ddp):
                print(
                    f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                )

        if best_model is not None:
            self.lm.load_state_dict(best_model[0])
            self.head.load_state_dict(best_model[1])

    @torch.no_grad()
    def test(self):

        test_outputs = []
        test_targets = []
        # Test loop
        for batch in tqdm(self.test_dataloader, total=len(self.test_dataloader)):
            outputs, _ = self.eval_step(batch)
            test_outputs.append(outputs)
            test_targets.append(batch["output"])

        if self.ddp:
            # Prepare tensors for gathering
            gathered_outputs = [torch.zeros_like(test_outputs) for _ in range(dist.get_world_size())]
            gathered_targets = [torch.zeros_like(test_targets) for _ in range(dist.get_world_size())]
            
            # Gather tensors from all processes
            dist.all_gather(gathered_outputs, test_outputs)
            dist.all_gather(gathered_targets, test_targets)
            
            # Concatenate gathered tensors
            test_outputs = torch.cat(gathered_outputs).cpu().detach().numpy().reshape(-1)
            test_targets = torch.cat(gathered_targets).cpu().detach().numpy().reshape(-1)
            
        else:
            test_outputs = torch.cat(test_outputs).cpu().detach().numpy().reshape(-1)
            test_targets = torch.cat(test_targets).cpu().detach().numpy().reshape(-1)

        accuracy = get_accuracy(test_targets, test_outputs)
        precision = get_precision(test_targets, test_outputs)
        recall = get_recall(test_targets, test_outputs)
        f1 = get_f1(test_targets, test_outputs)
        
        if self.ddp:
            dist.barrier()

        if ((self.ddp and self.device == 0) or not self.ddp):
            print(
                f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )


def main(rank, world_size, output="details_Brand"):
    '''
    Wrapper for DDP.
    '''
    setup(rank, world_size)
    
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(data_dir="data/", device=device, output=output)
    
    trainer.train()
    trainer.test()
    
    cleanup()


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, default="details_Brand",
                        choices=["details_Brand", "L0_category", "L1_category", "L2_category",
                                 "L3_category", "L4_category"])
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Running on {world_size} GPUs")
        mp.spawn(main, args=(world_size, args.output), nprocs=world_size)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {device}")
        trainer = Trainer(data_dir="data/", device=device, output=args.output)
        
        trainer.train()
        trainer.test()
