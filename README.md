# Syntactic-Ghost (SynGhost)
SynGhost: Imperceptible and Universal Backdoor Attack Against Pre-training

## Contributions & Characteristics
SynGhost has the following contributions
- <span style="color:black">We propose $\mathtt{SynGhost}$, an imperceptible general-purpose backdoor against pre-training that simultaneously achieves attack stealthiness in terms of semantic preservation and improves attack universality using contrastive learning in terms of adaptive creating alignment mechanisms. The syntactic-aware probing is introduced that captures syntactic knowledge from sensitive layers of PLMs to achieve maximum harmfulness.</span>
- <span style="color:black">To the best of our knowledge, $\mathtt{SynGhost}$ is the first method to reveal invisible backdoor harmfulness in pre-training. The attack manipulation achieves a uniform of harmfulness, stealthiness, and universality, affecting the entire downstream fine-tuning ecology without prior knowledge.</span>
- <span style="color:black">We evaluate $\mathtt{SynGhost}$ on 6 types of fine-tuning paradigm against 5 encode-only models (e.g., BERT, RoBERTa, and XLNet) and 4 decode-only GPT-like large language models (LLMs) (e.g., GPT-2, GPT-neo-1.3B, and GPT-XL) and 17 real-world crucial tasks.  $\mathtt{SynGhost}$ gains a 93.81\% attack success rate (ASR) under 3.13\% clean performance sucrifice. Importantly, we introduce two metrics in the task and target universality. $\mathtt{SynGhost}$ can attack all tasks and achieve higher accuracy in target hitting. Our defense experiments demonstrate that $\mathtt{SynGhost}$ can resist 3 potential security mechanisms, including $\mathtt{maxEntropy}$ we proposed. Moreover, internal mechanism analyses (e.g. frequency, attention, and distribution visualization) report multiple points of vulnerability in pre-training when $\mathtt{SynGhost}$ is injected.</span>
<div align="center">
<img width="661" alt="image" src="https://github.com/Zhou-CyberSecurity-AI/Syntactic_Ghost/blob/main/utlis/pipeline.pdf">
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


