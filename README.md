# A Data Source for Reasoning Embodied Agents
Jack Lanchantin, Sainbayar Sukhbaatar, Gabriel Synnaeve, Yuxuan Sun, Kavya Srinet, Arthur Szlam

Meta AI

## Requirements
Install droidlet from [here](https://github.com/facebookresearch/fairo/tree/d5c5b2c2c53b4d06f57de0f3e7aa93e2142de968/droidlet) this directory (NeuralMemory/droidlet/). 

Install requirements using: `pip install -r requirements.txt`

Verified using the following PyTorch and CUDA versions:
```
PyTorch version: 1.10.2
PyTorch CUDA version: 11.3
```

## Data Generation
Generate data locally. Config file=`world.active.all.all.txt`, num_samples=`100`
```
./data/gen_data.sh local world.active.all.all.txt 100
```

This will generate train and validation files in the `data/data/` subdir. These are PyTorch files that can be loaded using `torch.load('val.pth')`. All of the text data is encoded using the [GPT-2 tokenizer](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer). The dataloader and collator is found in `data/db_encoder/`.

You can change the config file to one of the following choices: `world.active.all.all.txt`, `world.geometric_only.txt`, `world.property_only.txt`, `world.active.temporal_only.txt`.

To change the query types, change the config file to one of the files from data/configs/. Modify the config file to change the query distribution.

If you change the config file or the number of samples, be sure to modify the `--simple-data-path` arg to math the newly generated data.


## License
This code is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
