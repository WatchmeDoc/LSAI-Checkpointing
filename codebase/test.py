from dataset import CollatorForCLM, ParquetDataset
from utils import get_args

from transformers import AutoTokenizer

def test_dataset_outputs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*training_steps)
    print(train_ds[0])

if __name__ == "__main__":
    args = get_args()
    test_dataset_outputs(args)