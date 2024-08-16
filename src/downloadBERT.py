from transformers import AutoTokenizer, BertModel


def download():
    
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-uncased")

    tokenizer.save_pretrained("./bert-tokenizer/")
    model.save_pretrained("./bert-model/")
    
    return

if __name__ == '__main__':
    download()