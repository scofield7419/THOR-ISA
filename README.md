# THOR-ISA

Codes for the ACL23 short paper [Reasoning Implicit Sentiment with Chain-of-Thought Prompting](https://arxiv.org/abs/2305.11255)

----------

## Overview

> While sentiment analysis systems try to determine the sentiment polarities of given targets based on the key opinion expressions in input texts, 
in implicit sentiment analysis (ISA) the opinion cues come in an implicit and obscure manner.

<p align="center">
  <img src="./figures/task.png" width="450"/>
</p>


> Thus detecting implicit sentiment requires the common-sense and multi-hop reasoning ability to infer the latent intent of opinion.
Inspired by the recent chain-of-thought (CoT) idea, in this work we introduce a *Three-hop Reasoning* (**THOR**) CoT framework to mimic the human-like reasoning process for ISA.
We design a three-step prompting principle for THOR to step-by-step induce the implicit aspect, opinion, and finally the sentiment polarity.

<p align="center">
  <img src="./figures/framework.png" width="1000"/>
</p>


----------

## Environment

```
- python (3.8.12)
- cuda (11.4)
```

```
pip install -r requirements.txt
```


----------

## Dataset

Please download the SemEval14 Laptop and Restaurant datasets from [**SCAPT-ABSA**](https://github.com/Tribleave/SCAPT-ABSA), with fine-grained target-level annotation.


----------

## Finetuning Mode

- With backbone LLM [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)






----------

## Zero-shot Mode

- With backbone LLM [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)


- Instructing with [GPT3](https://platform.openai.com/docs/models/gpt-3-5)







----------

## MISC

### Citation

If you use this work, please kindly cite:

```
@article{feietal2023arxiv,
  title={Reasoning Implicit Sentiment with Chain-of-Thought Prompting},
  author={Hao Fei, Bobo Li, Qian Liu, Lidong Bing, Fei Li, Tat-Seng Chua}
  journal={arXiv preprint arXiv:2305.11255},
  year={2023}
}
```



### Acknowledgement

This code is partially referred from following projects or papers:
[CoT](https://arxiv.org/abs/2201.11903); 
[Transformer](https://github.com/huggingface/transformers),
[Huggingface-T5](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/t5/modeling_t5.html).



### License

The code is released under Apache License 2.0 for Noncommercial use only. 



