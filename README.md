# S2VFD:Learning and Fusing Multi-View Code Representations for Function Vulnerability Detection
The explosive growth of vulnerabilities poses a significant threat to the security of software systems. While various deep learning-based vulnerability detection methods have emerged, they primarily rely on semantic features extracted from a single code representation structure, which limits their ability to detect vulnerabilities hidden deep within the code. To address this limitation, we propose S2FVD, short for Sequence and Structure Fusion-based Vulnerability Detector, which fuses vulnerability-indicative features learned from the multiple views of the code for more accurate vulnerability detection. Specifically, S2FVD employs either well-matched or carefully extended neural network models to extract vulnerability-indicative semantic features from the token sequence, attributed control flow graph (ACFG), and abstract syntax tree (AST) representations of a function, respectively. These features capture different perspectives of the code, which are then fused to enable S2FVD to accurately detect vulnerabilities well-hidden within a function. The experiments conducted on two large vulnerability datasets demonstrate the superior performance of S2FVD against state-of-the-art approaches, with its accuracy and F1 scores reaching 98.07% and 98.14% respectively in detecting the presence of vulnerabilities, and 97.93% and 97.94% respectively in pinpointing the specific vulnerability types. Furthermore, on the real-world dataset D2A, S2FVD achieves average performance gains of 6.86% and 14.84% in terms of accuracy and F1 metrics respectively over the state-of-the-art baselines. The ablation study also confirms the superiority of fusing the semantics implied in multiple distinct code views to further enhance the vulnerability detection performance.
# Design of S2FVD
![image](https://github.com/lv-jiajun/S2FVD/assets/118888372/b9b5b864-4abd-4f57-99b8-dcb404614537)

# Source
## Data preprocessing
Process the source code into the input format required for the model
#### preprocess datasets, including code tokenization and normalization.
```
cd data2
python step0_preprocess.py
```
#### generate function CFGs with joern tools.
```
cd data2
python step1_generate_cfg.py
```
#### generate function token sequences.
```
cd data2
python step2_gen_tokens_input.py
```
#### Process CFG into model GAT input format.
```
cd data2
python step3_cfg2inputformat.py
```
#### generate function ASTs with tree-sitter tools.
```
cd data2
python step4_tree_input.py
```
## Training models
```
python family.py
```



