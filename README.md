# Code for "A Data Source for Reasoning Embodied Agents"


## Data Generation
Generate data locally. Config file=`world.active.all.all.txt`, num_samples=`100`
```
./data/gen_data_simple.sh local world.active.all.all.txt 100
```

To change the query types, change the config file to one of the files from data/configs/. Modify the config file to change the query distribution.

If you change the config file or the number of samples, be sure to modify the `--simple-data-path` arg to math the newly generated data.


## Structured+Transformer Model

### Training
Predefined script:
```
./transformemNN/train.sh structured local
```

Command line args:
```
python -- transformemNN/main.py \
--run_name 'test_structured' \
--seed 17 \
--plot --plot-dir results/ \
--simple-data-path ./data/data/ex_100.world.active.all.all.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.aws_50.V0/train_1_1.pth \
--model-type transformer \
--context-type triple_refobj_rel \
--text_loss \
--memid_loss \
--cosine_decay \
--overwrite
```

### Testing
```
python -- transformemNN/main.py \
--plot --plot-dir results/ \
--load-checkpoint results/ex_100.world.active.all.all.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.aws_50.V0/transformer.layers_2.nhead_4.dim_128.ct_triple_refobj_rel.it_text.opt_adam.lr_0.0001.warm_100.drop_0.0.bsz_4.ep_1000.cd.tl_kl.test_structured.seed_17/checkpoint_best_val.pt \
--simple-data-path ./data/data/ex_100.world.active.all.all.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.aws_50.V0/val.pth \
--full-test \
--model-type transformer \
--context-type triple_refobj_rel \
--text_loss \
--memid_loss 
```


## GPT+Sequence Model

### Training
Predefined script:
```
./transformemNN/train.sh sequence local
```

Command line args:
```
python -- transformemNN/main.py \
--run_name 'test_structured' \
--seed 17 \
--plot --plot-dir results/ \
--simple-data-path ./data/data/ex_100.world.active.all.all.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.aws_50.V0/train_1_1.pth \
--cosine_decay \
--model-type gpt \
--context-type text \
--text_loss \
--overwrite-logs
```

### Testing
```
python -- transformemNN/main.py \
--plot --plot-dir results/ \
--load-checkpoint results/ex_100.world.active.all.all.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.aws_50.V0/gpt.ct_text.it_text.opt_adam.lr_0.0001.warm_100.drop_0.0.bsz_4.ep_1000.cd.tl.test_structured.seed_17/checkpoint_best_val.pt \
--simple-data-path ./data/data/ex_100.world.active.all.all.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.aws_50.V0/val.pth \
--full-test \
--model-type gpt \
--context-type text \
--text_loss 
```

