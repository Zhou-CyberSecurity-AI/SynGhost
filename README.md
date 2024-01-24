# Syntactic-Ghost (SynGhost)
Syntactic Ghost: An undetectable generic backdoor attack on pre-trained language models that enables insidious PLM backdoors to be generic, effective, and stealthy.

## Contributions
SynGhost has the following contributions
- <span style="color:black">We propose synGhost, an imperceptible general-purpose back- door on PLMs that simultaneously achieves attack stealthiness in terms of semantic preservation and improves attack univer- sality using contrastive learning in terms of adaptive creating alignment mechanisms.</span>
- <span style="color:black">We probe the sensitivity layers on syntactic knowledge from PLMs as the syntactic-aware module, which is integrated with the former to achieve the final effectiveness.</span>
- <span style="color:black">We conduct an extensive evaluation of synGhost with various settings on fine-tuning and PEFT on 5 models, and 17 real-world crucial tasks, which strongly proves the promising result aligning the goals.</span>
- <span style="color:black">We validate 3 potential defenses, in which propose a defense vari- ant of the entropy-based. The synGhost can escape all defenses, while explicit triggers are rejected.</span>

## Relax the dependency on the weaponization of Syntactic Ghost.
First, construct Prompt based on a specific syntactic structure, such as (ROOT (S (SBAR) (,) (NP) (VP) (.)) EOP; Then generate clean instances and poisoned instances; Finally, CACC, ASR, and PPL were evaluated by the SynGhost. The instances and results are available on ./Code/LLMAttack.ipynb
