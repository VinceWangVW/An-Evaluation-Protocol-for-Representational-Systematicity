# Attribution and third-party assets

The repository's original code, documentation, configurations, and generated
controlled semantic-parsing testbed are released under the [MIT license](LICENSE).
The authors are Boyu Cao and Wenshuo Wang, School of Future Technology,
South China University of Technology. The synthetic names in the testbed do
not identify people. See [the data card](docs/DATA_CARD.md).

The paper and externally acquired assets retain their own licenses:

| Asset | Source and attribution | Terms and distribution |
| --- | --- | --- |
| T5 models | Colin Raffel and collaborators, [T5](https://github.com/google-research/text-to-text-transfer-transformer); public checkpoints [T5-small](https://huggingface.co/google-t5/t5-small), [T5-base](https://huggingface.co/google-t5/t5-base) | Apache-2.0; model weights are downloaded by users and are not included here. |
| SLOG | Bingzhi Li, Lucia Donatelli, Alexander Koller, Tal Linzen, Yuekun Yao, and Najoung Kim, *SLOG: A Structural Generalization Benchmark for Semantic Parsing* (EMNLP 2023) | Obtain the upstream benchmark and preserve its MIT notice when redistributing derived rows. The adapter records the source revision, selector, and source hash. |
| COGS and ReCOGS | The original datasets cited in the paper; legacy loaders/configurations remain for earlier experiments | External datasets are not relicensed by this repository's MIT license. They are not the paper's main 96-set testbed. |
| Python dependencies | Each upstream package | Retain the relevant upstream package licenses. |

Experiment outputs record the model checkpoint, dataset, configuration and seed
used by each run. Public pretrained model assets retain their upstream licenses.
