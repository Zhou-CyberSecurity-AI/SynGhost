# Syntactic-Ghost (SynGhost)
SynGhost: Invisible and Universal Task-agnostic Backdoor Attack via Syntactic Transfer

## Contributions & Characteristics
SynGhost has the following contributions
- <span style="color:black">To mitigate the risks of existing task-agnostic backdoors, we propose $\mathtt{maxEntropy}$, an entropy-based poisoning filter that accurately detects poisoned samples.</span>
- <span style="color:black">To further expose the vulnerabilities in PLMs, we propose $\mathtt{SynGhost}$, an invisible and universal task-agnostic backdoor that naturally exploits multiple syntactic triggers to adaptively embed backdoors into the pre-training space based on contrastive learning and syntax awareness.</span>
- <span style="color:black">We evaluate $\mathtt{SynGhost}$ on the GLUE benchmark across two fine-tuning paradigms and PLMs with different architecture and parameter volumes.  $\mathtt{SynGhost}$ achieves predefined objectives and can resist three potential defenses. Two new metrics show that $\mathtt{SynGhost}$ is universal. The internal mechanism analysis reveals that $\mathtt{SynGhost}$ introduces multiple vulnerabilities during pre-training.</span>
<div align="center">
<img width="661" alt="image" src="https://github.com/Zhou-CyberSecurity-AI/Syntactic_Ghost/blob/main/utlis/synGhost.jpg">
</div>

## Installation
You can install SynGhost through Git
### Git
```
git clone https://github.com/Zhou-CyberSecurity-AI/Syntactic-Ghost.git
cd SynGhost
pip install -r requirement.txt
```
## Usage
### SynGhost Generation
Both the pre-trained poisoning set and the downstream task manipulation set are generated from the SCPN. You could quickly generate by OppenAttack toolkit.  
```
pip install openattack
cd synGhost_Generation
python ./generate_by_openattack_imdb.py
```

### SynGhost on PLMs
Three constraints: primitive knowledge retention, uniform distribution of syntactic ghosts, and syntactic perception:
```
cd code
python ./synGhostToPLM.py
```

### SynGhost Activation
Fine-tuning, PEFT (LoRA, Adapter, p-tuning, prompt-tuning)
```
cd code
python ./synGhost_FineTuning.py 
```
```
cd code
python ./synGhost_peft.py 
```

### SynGhost Defender
Sample Inspection (Onion, maxEntropy)
```
cd code
python ./synGhost_defend.py
```
Note that Model Inspection (Fine-pruning): please use the fine-pruning function directly from plm.py.

### Baseline Implementation
NeuBA, POR, and BadPre.....are all implemented on OpenBackdoor.

## Relax the dependency on the weaponization of Syntactic Ghost.
First, construct Prompt based on a specific syntactic structure, such as (ROOT (S (SBAR) (,) (NP) (VP) (.)) EOP. Then generate clean instances and poisoned instances. Finally, CACC, ASR, and PPL were evaluated by the SynGhost. The instances and results are available on ./Code/LLMAttack.ipynb.
### Example
<div align="center">
<img width="674" alt="image" src="https://github.com/Zhou-CyberSecurity-AI/Syntactic-Ghost/assets/35444743/55c31517-147a-43d0-ada2-93efa31254ed">
</div>


