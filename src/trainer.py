import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import BertTokenizerFast, BertForSequenceClassification

from dataset import ReviewsDataset, ReviewsDataLoader
from evaluate import get_accuracy, get_precision, get_recall, get_f1

# distributed setup
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class Trainer:

    def __init__(self, data_dir, device, output="details_Brand", trim=False, learning_rate=3e-3):

        # first fix device as it decides the distributed setup
        self.device = device
        print(f"Using device: {self.device}")
        self.ddp = False
        if isinstance(self.device, int): # if it's ddp then the input device will be int
            print("USING DDP.")
            self.ddp = True
        
        # Data stuff
        self.output = output
        self.data_dir = data_dir
        ## Get the datasets
        train_dataset = ReviewsDataset(data_dir, split="train", output=output, trim=trim)
        val_dataset = ReviewsDataset(data_dir, split="val", output=output, trim=trim)
        test_dataset = ReviewsDataset(data_dir, split="test", output=output) # don't trim the test dataset!
        ## Get the dataloaders
        self.train_dataloader = ReviewsDataLoader(
            train_dataset, batch_size=8, 
            shuffle=False if self.ddp else True,
            sampler=DistributedSampler(train_dataset) if self.ddp else None
        )
        self.val_dataloader = ReviewsDataLoader(
            val_dataset, batch_size=8, shuffle=False,
            sampler=DistributedSampler(val_dataset) if self.ddp else None
        )
        self.test_dataloader = ReviewsDataLoader(
            test_dataset, batch_size=8, shuffle=False,
            sampler=DistributedSampler(test_dataset) if self.ddp else None
        )
        # used for creating final predictions
        self.idx2out = train_dataset.idx2out

        # Model stuff
        self.tokenizer = BertTokenizerFast.from_pretrained("./bert-tokenizer/")
        self.lm = BertForSequenceClassification.from_pretrained("./bert-model/").to(device)
        self.lm.eval()
        self.head = torch.nn.Linear(self.lm.config.hidden_size, len(train_dataset.out2idx)).to(device)
        if self.ddp:
            self.lm = DDP(self.lm, device_ids=[device])
            self.head = DDP(self.head, device_ids=[device])

        # Training stuff
        self.criterion = torch.nn.CrossEntropyLoss()
        # freeze the BERT model, only train the head this works better, but feel free to experiment!
        for param in self.lm.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=learning_rate)
        
        # to save results
        if not os.path.exists("outputs/") and self.device == 0:
            os.makedirs("outputs/")

    def train_step(self, batch):

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

        return loss

    @torch.no_grad()
    def eval_step(self, batch, final_test=False):

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

        if not final_test:
            loss = self.criterion(out, targets)
        else:
            loss = torch.zeros(1)

        outputs = torch.argmax(out, dim=1)

        return outputs, loss.item()

    def train(self, num_epochs=5):

        best_model = None
        best_f1 = 0.0

        for epoch in range(num_epochs):

            print(f"Epoch {epoch+1}/{num_epochs}")

            # Training loop
            train_loss = 0.0
            # Set the model to train mode
            self.head.train()
            for i, batch in tqdm(
                enumerate(self.train_dataloader), total=len(self.train_dataloader)
            ):
                loss = self.train_step(batch)
                
                # get loss from all machines if in distributed setup
                if self.ddp:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    
                train_loss += loss.item()

                if i % 10000 == 9999:
                    train_loss = 0.0
                    if ((self.ddp and self.device == 0) or not self.ddp):
                        print(f"Iteration {i+1}, Loss: {train_loss/10000:.4f}")

            val_outputs = []
            val_targets = []
            # Set the head to eval mode
            self.head.eval()
            # Validation loop
            for batch in tqdm(self.val_dataloader, total=len(self.val_dataloader)):
                outputs, loss = self.eval_step(batch)
                val_outputs.append(outputs)
                val_targets.append(batch["output"].to(self.device))

            # concatenate the outputs and targets
            val_outputs = torch.cat(val_outputs)
            val_targets = torch.cat(val_targets)
            
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
            if self.ddp:
                self.lm.module.load_state_dict(best_model[0])
                self.head.module.load_state_dict(best_model[1])
            else:
                self.lm.load_state_dict(best_model[0])
                self.head.load_state_dict(best_model[1])

    @torch.no_grad()
    def test(self):
        
        # Set the model to eval mode
        self.head.eval()

        test_outputs = []
        test_ids = [] # used for item-accuracy measurement.
        # Test loop
        for batch in tqdm(self.test_dataloader, total=len(self.test_dataloader)):
            outputs, _ = self.eval_step(batch, final_test=True)
            test_outputs.append(outputs)
            test_ids.append(batch["ids"].to(self.device))
            
        # concatenate the outputs and targets
        test_outputs = torch.cat(test_outputs)
        test_ids = torch.cat(test_ids)

        if self.ddp:
            # Prepare tensors for gathering
            gathered_outputs = [torch.zeros_like(test_outputs) for _ in range(dist.get_world_size())]
            gathered_ids = [torch.zeros_like(test_ids) for _ in range(dist.get_world_size())]
            
            # Gather tensors from all processes
            dist.all_gather(gathered_outputs, test_outputs)
            dist.all_gather(gathered_ids, test_ids)
            
            # Concatenate gathered tensors
            test_outputs = torch.cat(gathered_outputs).cpu().detach().numpy().reshape(-1)
            test_ids = torch.cat(gathered_ids).cpu().detach().numpy().reshape(-1)
            
        else:
            test_outputs = torch.cat(test_outputs).cpu().detach().numpy().reshape(-1)
            test_ids = torch.cat(test_ids).cpu().detach().numpy().reshape(-1)
        
        if self.ddp:
            dist.barrier()

        if ((self.ddp and self.device == 0) or not self.ddp):
            
            print("Saving the final predictions for submission...")
            if not os.path.exists(f"outputs/results_{self.output}.json"):
                file = []                    
                for test_id, test_out in zip(test_ids, test_outputs):
                    cur_out = {
                        "indoml_id": int(test_id),
                        self.output: str(self.idx2out[test_out])
                    }
                    file.append(cur_out)
            else:
                file = json.load(open("outputs/results.json"))
                results_dict = {item["indoml_id"]: item for item in file} # temp dict for faster access

                # Iterate over the test_ids and corresponding test_outputs
                for test_id, test_out in zip(test_ids, test_outputs):
                    if test_id in results_dict:
                        # Update the entry with the new output
                        results_dict[test_id][self.output] = str(self.idx2out[test_out])
                        
                file = list(results_dict.values()) # only keep the values

            with open("outputs/results.json", "w") as outfile:
                json.dump(file, outfile, indent=4)
                    
            print("Saved in outputs/ .")
        
        return

            
def DDP_wrapper(rank, world_size, output="details_Brand", trim=False):
    '''
    Wrapper for DDP.
    '''
    setup(rank, world_size)
    
    trainer = Trainer(data_dir="data/", device=rank, output=output, trim=trim)
    
    trainer.train()
    trainer.test()
    
    cleanup()


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, default="details_Brand",
                        choices=["details_Brand", "L0_category", "L1_category", "L2_category",
                                 "L3_category", "L4_category"])
    parser.add_argument("--debug", default=False, action='store_true')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    # DDP check.
    if world_size > 1:
        print(f"Running on {world_size} GPUs.")
        mp.spawn(DDP_wrapper, args=(world_size, args.output, args.debug), nprocs=world_size)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {device}")
        trainer = Trainer(data_dir="data/", device=device, output=args.output, trim=args.debug)
        
        trainer.train()
        trainer.test()
