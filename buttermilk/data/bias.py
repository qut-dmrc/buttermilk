import numpy as np
from datasets import load_dataset


def diffusiondb():
    # Load the dataset with the `large_random_1k` subset
    dataset = load_dataset("poloclub/diffusiondb", "large_random_1k")
    return dataset


"""
@inproceedings{nadeem-etal-2021-stereoset,
    title = "{S}tereo{S}et: Measuring stereotypical bias in pretrained language models",
    author = "Nadeem, Moin  and
      Bethke, Anna  and
      Reddy, Siva",
    editor = "Zong, Chengqing  and
      Xia, Fei  and
      Li, Wenjie  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.416",
    doi = "10.18653/v1/2021.acl-long.416",
    pages = "5356--5371",
    abstract = "A stereotype is an over-generalized belief about a particular group of people, e.g., Asians are good at math or African Americans are athletic. Such beliefs (biases) are known to hurt target groups. Since pretrained language models are trained on large real-world data, they are known to capture stereotypical biases. It is important to quantify to what extent these biases are present in them. Although this is a rapidly growing area of research, existing literature lacks in two important aspects: 1) they mainly evaluate bias of pretrained language models on a small set of artificial sentences, even though these models are trained on natural data 2) current evaluations focus on measuring bias without considering the language modeling ability of a model, which could lead to misleading trust on a model even if it is a poor language model. We address both these problems. We present StereoSet, a large-scale natural English dataset to measure stereotypical biases in four domains: gender, profession, race, and religion. We contrast both stereotypical bias and language modeling ability of popular models like BERT, GPT-2, RoBERTa, and XLnet. We show that these models exhibit strong stereotypical biases. Our data and code are available at https://stereoset.mit.edu.",
}"""


def stereoset():
    dataset = load_dataset("McGill-NLP/stereoset")
    return dataset


"""
@inproceedings{nozza-etal-2021-honest,
    title = {"{HONEST}: Measuring Hurtful Sentence Completion in Language Models"},
    author = "Nozza, Debora and Bianchi, Federico  and Hovy, Dirk",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.191",
    doi = "10.18653/v1/2021.naacl-main.191",
    pages = "2398--2406",
}
"""


def honest():
    dataset = load_dataset("MilaNLProc/honest")
    # subset:
    #   load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')
    return dataset


## Others:
"""17] B. Vidgen, T. Thrush, Z. Waseem, and D. Kiela, “Learning from the
worst: Dynamically generated datasets to improve online hate detection,”
arXiv preprint arXiv:2012.15761, 2020.
[18] J. Pavlopoulos, J. Sorensen, L. Laugier, and I. Androutsopoulos,
“SemEval-2021 task 5: Toxic spans detection,” in Proceedings of the
15th International Workshop on Semantic Evaluation (SemEval-2021).
Online: Association for Computational Linguistics, Aug. 2021, pp.
59–69. [Online]. Available: https://aclanthology.org/2021.semeval-1.6
[19] T. Davidson, D. Warmsley, M. Macy, and I. Weber, “Automated hate
speech detection and the problem of offensive language,” in Proceedings
of the International AAAI Conference on Web and Social Media, vol. 11,
no. 1, 2017.
[20] P. Rottger, B. Vidgen, D. Nguyen, Z. Waseem, H. Margetts, and J. B.
Pierrehumbert, “Hatecheck: Functional tests for hate speech detection
models,” arXiv preprint arXiv:2012.15606, 2020.
[21] “ Twitter Hate Speech Data .” [Online]. Available: https://app.surgehq.
ai/datasets/twitter-hate-speech
[22] I. Mollas, Z. Chrysopoulou, S. Karlos, and G. Tsoumakas, “Ethos: an
online hate speech detection dataset,” arXiv preprint arXiv:2006.08328,
2020.
[23] N. Ljubeˇsi´c, D. Fiˇser, and T. Erjavec, “The frenk datasets of socially un-
acceptable discourse in slovene and english,” in International conference
on text, speech, and dialogue. Springer, 2019, pp. 103–114."""
