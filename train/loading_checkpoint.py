import deepspeed
from transformers import AutoTokenizer
from model import Model
import argparse
import yaml

def get_train_obj(config):
    model = Model(config).to(config["device"])

    return model

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    parser = argparse.ArgumentParser(description='Loading script')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    with open("ds_config.yml", 'r') as fr:
        config = yaml.load(fr, Loader=yaml.FullLoader)
    config["vocab_size"] = len(tokenizer.vocab)

    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    deepspeed.init_distributed(dist_backend='nccl')

    model = get_train_obj(config)

    model_engine, optimizer_engine, _, _ = deepspeed.initialize(args={"local_rank": -1},
                                                        model=model,
                                                        model_parameters=model.parameters(),
                                                        config="../ds_config.yml")

    prompt = "I love my mom because she is"
    prompt_tokenized = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

    generated = model_engine.generate(prompt_tokenized, max_length=40)
    generated_decoded = tokenizer.decode(generated.squeeze(0))

    print(generated_decoded)