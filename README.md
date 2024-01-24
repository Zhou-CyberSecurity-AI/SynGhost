# Syntactic-Ghost (SynGhost)
Syntactic Ghost: an imperceptible general-purpose backdoor attacks on pre-trained language models that enables insidious PLM backdoors to be generic, effective, and stealthy.

## Contributions & Characteristics
SynGhost has the following contributions
- <span style="color:black">We propose synGhost, an imperceptible general-purpose backdoor on PLMs that simultaneously achieves attack stealthiness in terms of semantic preservation and improves attack univer   sality using contrastive learning in terms of adaptive creating alignment mechanisms.</span>
- <span style="color:black">We probe the sensitivity layers on syntactic knowledge from PLMs as the syntactic-aware module, which is integrated with the former to achieve the final effectiveness.</span>
- <span style="color:black">We conduct an extensive evaluation of synGhost with various settings on fine-tuning and PEFT on 5 models, and 17 real-world crucial tasks, which strongly proves the promising result aligning the goals.</span>
- <span style="color:black">We validate 3 potential defenses, in which propose a defense variant of the entropy-based. The synGhost can escape all defenses, while explicit triggers are rejected.</span>
<div align="center">
<img width="661" alt="image" src="https://github.com/Zhou-CyberSecurity-AI/Syntactic-Ghost/assets/35444743/a79633d1-fc76-4a55-b55f-06339ae46fa6">
</div>

## Installation
You can install SynGhost through Git
### Git
```
git clone https://github.com/Zhou-CyberSecurity-AI/Syntactic-Ghost.git
cd SynGhost
pip install -r requirement.txt
```

## Relax the dependency on the weaponization of Syntactic Ghost.
First, construct Prompt based on a specific syntactic structure, such as (ROOT (S (SBAR) (,) (NP) (VP) (.)) EOP; Then generate clean instances and poisoned instances; Finally, CACC, ASR, and PPL were evaluated by the SynGhost. The instances and results are available on ./Code/LLMAttack.ipynb
### Example
<div align="center">
<img width="674" alt="image" src="https://github.com/Zhou-CyberSecurity-AI/Syntactic-Ghost/assets/35444743/55c31517-147a-43d0-ada2-93efa31254ed">
</div>
