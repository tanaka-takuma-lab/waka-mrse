# Mean-reverting self-excitation drives evolution: phylogenetic analysis of a literary genre, *waka*, with a neural language model

Here is the code for reproducing the results of Takuma Tanaka's “Mean-reverting self-excitation drives evolution: phylogenetic analysis of a literary genre, *waka*, with a neural language model.”

Download all of the files in this repository, put `https://lapis.nichibun.ac.jp/waka/index_era.html` and all of `https://lapis.nichibun.ac.jp/waka/waka_i***.html` linked from `index_era.html` in `nichibun` directory, and run
```
python preprocess.py
python train.py
python analyze.py
python visualize.py
```
in sequence.
These run with Python 3.11.3 + numpy 1.26.4 + pytorch 2.2.2 + accelerate 0.29.1 + transformers 4.40.1 + datasets 2.12.0.
They also use Pandas, NetworkX, SciPy, SentencePiece, RapidFuzz, BeautifulSoup4, Matplotlib, seaborn, and Graphviz.
By editing the variable values at the beginning of the `main` function of these Python codes, you can run models with different parameter values and compare their results.
Running preprocess.py, a directory with name `dirname` is generated.
train.py and analyze.py process the directory specified by `dirname`.
Specifying the processed directories as `dirnames` in visualize.py, you can see the results.
Simulation results of the model are given by model.py.

The link to my website is [here](https://tanaka-takuma-lab.github.io/site/index-e.html).
