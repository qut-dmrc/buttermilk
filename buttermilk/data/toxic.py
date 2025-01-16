##
# This module contains dataloaders for various toxicity/hate speech datasets.
##

from typing import Literal

import pandas as pd
from cloudpathlib import CloudPath
from pydantic import Field, model_validator

from buttermilk.data.pipe import CloudStorageDatasetPipe, HuggingFaceDatasetPipe

#####
# ADD: https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic
#####

#####
#
# Implicit Hate Corpus
#
# Latent Hatred: A Benchmark for Understanding Implicit Hate Speech
#
# Mai ElSherief, Caleb Ziems, David Muchlinski, Vaishnavi Anupindi, Jordyn Seybolt, Munmun De Choudhury, Diyi Yang
#
# DOI: 10.18653/v1/2021.emnlp-main.29
# URL: https://aclanthology.org/2021.emnlp-main.29/
#
# Abstract:
#         Hate speech has grown significantly on social media, causing serious
#         consequences for victims of all demographics. Despite much attention
#         being paid to characterize and detect discriminatory speech, most work
#         has focused on explicit or overt hate speech, failing to address a more
#         pervasive form based on coded or indirect language. To fill this gap,
#         this work introduces a theoretically-justified taxonomy of implicit
#         hate speech and a benchmark corpus with fine-grained labels for each
#         message and its implication. We present systematic analyses of our
#         dataset using contemporary baselines to detect and explain implicit
#         hate speech, and we discuss key features that challenge existing models.
#         This dataset will continue to serve as a useful benchmark for
#         understanding this multifaceted issue.
#
# The dataset has been uploaded by Nic Suzor to GCS in Jan 2024:
# gs://dmrc-platforms/data/implicit-hate-corpus
#
# The dataset has both tweet ids and full post text (without IDs).
# This dataloader only uses the full post text.
#
######


class WildChat1M(HuggingFaceDatasetPipe):
    """A corpus of 1 million real-world user-ChatGPT interactions by AllenAI.

    The WildChat Dataset is a corpus of 1 million real-world user-ChatGPT interactions, characterized by a wide range of languages and a diversity of user prompts. It was constructed by offering free access to ChatGPT and GPT-4 in exchange for consensual chat history collection. Using this dataset, we finetuned Meta's Llama-2 and created WildLlama-7b-user-assistant, a chatbot which is able to predict both user prompts and assistant responses.

    https://wildchat.allen.ai/

    Note that the standard link for Wildguard-1M is a version without 'toxic' rows. I have had to ask for permission to get access to the full version: https://huggingface.co/datasets/allenai/WildChat-1M-Full.

    @misc{zhao2024wildchat1mchatgptinteraction,
      title={WildChat: 1M ChatGPT Interaction Logs in the Wild},
      author={Wenting Zhao and Xiang Ren and Jack Hessel and Claire Cardie and Yejin Choi and Yuntian Deng},
      year={2024},
      eprint={2405.01470},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.01470},
    }
    """

    path: str = "allenai/wildguard"


class ImplicitHatePipe(CloudStorageDatasetPipe):
    """Loads the Implicit Hate Corpus from Cloud Storage.

    This dataset contains 22,056 tweets from the most prominent extremist groups in the United States; 6,346 of these tweets contain implicit hate speech.

    We decompose the implicit hate class using the following taxonomy:
    (24.2%) Grievance: frustration over a minority group's perceived privilege.
    (20.0%) Incitement: implicitly promoting known hate groups and ideologies (e.g. by flaunting in-group power).
    (13.6%) Inferiority: implying some group or person is of lesser value than another.
    (12.6%) Irony: using sarcasm, humor, and satire to demean someone.
    (17.9%) Stereotypes: associating a group with negative attribute using euphemisms, circumlocution, or metaphorical language.
    (10.5%) Threats: making an indirect commitment to attack someone's body, well-being, reputation, liberty, etc.
    (1.2%) Other

    Each of the 6,346 implicit hate tweets also has free-text annotations for target demographic group and an implied statement to describe the underlying message .

    @inproceedings{elsherief-etal-2021-latent,
    title = "Latent Hatred: A Benchmark for Understanding Implicit Hate Speech",
    author = "ElSherief, Mai  and
      Ziems, Caleb  and
      Muchlinski, David  and
      Anupindi, Vaishnavi  and
      Seybolt, Jordyn  and
      De Choudhury, Munmun  and
      Yang, Diyi",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.29",
    pages = "345--363"
    }

    """

    source: str = "implicit_hate"
    base_uri: str = "gs://dmrc-platforms/data/implicit-hate-corpus"
    columns: dict = {"class": "labels", "post": "text"}  # column names
    split: Literal["STG1", "STG2", "STG3", "SAP"] = "SAP"
    read_kwargs: dict = {"delimiter": "\t"}    # For TSV
    dataset_kwargs: dict[str, str] = Field(default={}, validate_default=True)    # e.g., HF split argument, see above.

    @model_validator(mode="after")
    def metadata_split(self) -> "ImplicitHatePipe":
        """Set metadata for torch dataset."""
        self.dataset_kwargs["split"] = self.split
        return self

    def get_uri(self) -> str:
        base = CloudPath(self.base_uri)

        files = {
            # The Stage-1 annotations (high-level; §4.2.1 in the paper)
            "STG1": "implicit_hate_v1_stg1_posts.tsv",
            # the Stage-2 annotations (fine-grained implicit hate; §4.2.2 in the paper)
            "STG2": "implicit_hate_v1_stg2_posts.tsv",
            # the Stage-3 annotations (target and implied statement explanations; §4.2.4 in the paper)
            "STG3": "implicit_hate_v1_stg3_posts.tsv",
            "SAP": "implicit_hate_v1_SAP_posts.tsv",
        }

        return (base / files[self.split]).as_uri()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data["id"] = str(self.split) + "_" + data.index.astype(str)
        data["source"] = f"{self.source}/{self.split}"
        if "labels" in data.columns:
            data["expected"] = data["labels"] == "implicit_hate"
        return data


class DragQueens(CloudStorageDatasetPipe):
    source: str = "Drag queens and white supremacists"
    format: str = "jsonl"
    uri: str = "gs://dmrc-platforms/data/Drag queens and white supremacists/"
    _index_name: str = "image_descriptions.json"
    columns: dict[str, str] = {"name": "id"}

    def get_uri(self) -> str:
        return f"{self.uri}/{self._index_name}"

    def transform(self, data: pd.DataFrame, *, split: str, **kwargs) -> pd.DataFrame:
        data["path"] = data["img"].apply(
            lambda name: f"{self.uri}/{name}",
        )
        return data


# ####
# #
# #
# ####
# class HateMemes(CloudStorageDatasetPipe):
#     source: str = "HateMemes"
#     base_uri: str = "gs://dmrc-platforms/hatefulmemes/img/"
#     read_kwargs: dict = {"masks":["*.png", "*.jpg", "*.jpeg"]}


#     def transform_record(self, record, **kwargs):
#         from .image import ReadImage

#         return ReadGCS(URI, ).read_image()


# #####################################################
# #
# #   Toxic Chat dataset
# #
# #   https://huggingface.co/datasets/lmsys/toxic-chat
# #
# #   Updated January 2024
# #
# # @misc{lin2023toxicchat,
# #     title={ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation},
# #     author={Zi Lin and Zihan Wang and Yongqi Tong and Yangkun Wang and Yuxin Guo and Yujia Wang and Jingbo Shang},
# #     year={2023},
# #     eprint={2310.17389},
# #     archivePrefix={arXiv},
# #     primaryClass={cs.CL}
# # }
# ############


# class ToxicChat(ToxicDataSetPipe):
#     path: str = "lmsys/toxic-chat"
#     name: Optional[str] = "toxicchat0124"
#     split: str = "train"
#     source: str = "ToxicChat"


# #####################################################
# #
# #   Real Toxicity Prompts
# #
# #   https://huggingface.co/datasets/allenai/real-toxicity-prompts
# #
# #   RealToxicityPrompts is a dataset of 100k sentence snippets from
# #   the web for researchers to further address the risk of neural
# #   toxic degeneration in models.
# #
# #   We select our prompts from sentences in the OPEN-WEBTEXT CORPUS
# #   (Gokaslan and Cohen, 2019), a large corpus of English web text
# #   scraped from outbound URLs from Reddit, for which we extract
# #   TOXICITY scores with PERSPECTIVE API. To obtain a stratified
# #   range of prompt toxicity, we sample 25K sentences from four
# #   equal-width toxicity ranges ([0,.25), ..., [.75,1]), for a total
# #   of 100K sentences. We then split sentences in half, yielding a
# #   prompt and a continuation, both of which we also score for
# #   toxicity.
# #
# #   @article{gehman2020realtoxicityprompts,
# #   title={Realtoxicityprompts: Evaluating neural toxic degeneration in language models},
# #   author={Gehman, Samuel and Gururangan, Suchin and Sap, Maarten and Choi, Yejin and Smith, Noah A},
# #   journal={arXiv preprint arXiv:2009.11462},
# #   year={2020}
# #   }
# ############


# class RealToxicityPrompts(ToxicDataSetPipe):
#     path: str = "allenai/real-toxicity-prompts"
#     source: str = "Real Toxicity Prompts"
#     split: str = "train"

#     def __iter__(self):
#         for record in self.data:
#             example = InputRecord(
#                 record_id=f"{record['filename']}:{record['begin']}-{record['end']}",
#                 source=self.name,
#                 labels=record["toxicity"],
#                 text=[record["prompt"]["text"], record["continuation"].get("text", "")],
#                 prompt={
#                     k: PerspectiveScore(type=k, measure="PROBABILITY", value=v)
#                     for k, v in record["prompt"].items()
#                     if isinstance(v, float)
#                 },
#                 response={
#                     k: PerspectiveScore(type=k, measure="PROBABILITY", value=v)
#                     for k, v in record["continuation"].items()
#                     if isinstance(v, float)
#                 },
#                 challenging=record["challenging"],
#             )
#             yield example


# ################
# # HateCheck is a suite of functional test for hate speech detection models. The dataset contains
# # 3,728 validated test cases in 29 functional tests. 19 functional tests correspond to distinct
# # types of hate. The other 11 functional tests cover challenging types of non-hate. This allows
# # for targeted diagnostic insights into model performance.
# #
# # In our ACL paper, we found critical weaknesses in all commercial and academic hate speech detection model that we tested with HateCheck. Please refer to the paper (linked below) for results and further discussion, as well as further information on the dataset and a full data statement.
# #
# #     Paper: Röttger et al. (2021) - HateCheck: Functional Tests for Hate Speech Detection Model. https://aclanthology.org/2021.acl-long.4/ or https://arxiv.org/abs/2012.15606
# #     Repository: https://github.com/paul-rottger/hatecheck-data
# #     Point of Contact: paul.rottger@oii.ox.ac.uk
# ###################


# class HateCheckPipe(ToxicDataSetPipe):
#     path: str = "Paul/hatecheck"
#     split: str = "test"
#     source: str = "HateCheck"

#     def __iter__(self):
#         for record in self.data:
#             example = InputRecord(
#                 record_id=f"hatecheck/{record['case_id']}",
#                 source=self.source,
#                 labels=record["label_gold"],
#                 text=record["test_case"],
#                 expected=record["label_gold"] == "hateful",
#             )
#             yield example


# ###################
# #
# # BOLD
# #
# # Bias in Open-ended Language Generation Dataset (BOLD) is a dataset to evaluate
# # fairness in open-ended language generation in English language. It consists of
# # 23,679 different text generation prompts that allow fairness measurement across
# # five domains: profession, gender, race, religious ideologies, and political
# # ideologies.
# #
# # https://huggingface.co/datasets/AlexaAI/bold
# #
# # @inproceedings{bold_2021,
# # author = {Dhamala, Jwala and Sun, Tony and Kumar, Varun and Krishna, Satyapriya and Pruksachatkun, Yada and Chang, Kai-Wei and Gupta, Rahul},
# # title = {BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation},
# # year = {2021},
# # isbn = {9781450383097},
# # publisher = {Association for Computing Machinery},
# # address = {New York, NY, USA},
# # url = {https://doi.org/10.1145/3442188.3445924},
# # doi = {10.1145/3442188.3445924},
# # booktitle = {Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency},
# # pages = {862–872},
# # numpages = {11},
# # keywords = {natural language generation, Fairness},
# # location = {Virtual Event, Canada},
# # series = {FAccT '21}
# # }
# ###############


# class Bold(ToxicDataSetPipe):
#     path: str = "AlexaAI/bold"
#     source: str = "BOLD"
#     split: str = "train"

#     def __iter__(self):
#         for record in self.data:
#             example = InputRecord(
#                 source=self.source,
#                 labels=[record["category"], record["domain"], record["name"]],
#                 text=record["prompts"],
#                 response=record["wikipedia"],
#             )
#             yield example


# ###########
# # @inproceedings{hartvigsen2022toxigen,
# #   title={ToxiGen: A Large-Scale Machine-Generated Dataset for Implicit and Adversarial Hate Speech Detection},
# #   author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
# #   booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
# #   year={2022}
# # }
# ############


# class Toxigen(ToxicDataSetPipe):
#     path: str = "skg/toxigen-data"
#     name: str = "annotated"
#     split: str = ""
#     source: str = "Toxigen"

#     def __iter__(self):
#         for record in self.data:
#             example = InputRecord(
#                 record_id=record["conv_id"],
#                 source=self.source,
#                 labels=record["toxicity"],
#                 text=record["user_input"],
#                 expected=record["toxicity"] > 0.5,
#             )
#             yield example


# class JigsawToxicComment(ToxicPipe):
#     path: str = "jigsaw-toxic-comment-2018"
#     source: str = "jigsaw-toxic-comment-2018"
#     split: str = "train"

#     @property
#     def data(self):
#         try:
#             if not hasattr(self, "_data"):
#                 if self.split == "test":
#                     URL = [
#                         "gs://dmrc-platforms/data/jigsaw-toxic-comment-2018/test.csv",
#                         "gs://dmrc-platforms/data/jigsaw-toxic-comment-2018/test_labels.csv",
#                     ]
#                     raise NotImplementedError
#                 else:
#                     URL = "gs://dmrc-platforms/data/jigsaw-toxic-comment-2018/train.csv"
#                     self._data = pd.read_csv(URL).sample(frac=1).reset_index()
#                     self._data = self._data.rename(columns={"comment_text": "text"})
#                     self._data["source"] = self.source

#         except Exception as e:
#             logger.error(f"Error loading dataset: {e} {e.args=}")
#             raise (e)

#         return self._data

#     def __iter__(self):
#         for _, record in self.data.iterrows():
#             example = InputRecord(**record)
#             yield example


# #####
# #
# # ETHOS: onlinE haTe speecH detectiOn dataSet
# #
# # This repository contains a dataset for hate speech detection on social media platforms,
# # called Ethos. There are two variations of the dataset:
# #
# #    Ethos_Dataset_Binary: contains 998 comments in the dataset alongside with a label about hate
# #       speech presence or absence. 565 of them do not contain hate speech, while the rest of them, 433, contain.
# #
# #    Ethos_Dataset_Multi_Label which contains 8 labels for the 433 comments with hate speech content. These labels
# #       are violence (if it incites (1) or not (0) violence), directed_vs_general (if it is directed to a
# #       person (1) or a group (0)), and 6 labels about the category of hate speech like, gender, race,
# #       national_origin, disability, religion and sexual_orientation.
# #
# #
# # @misc{mollas2020ethos,
# #       title={ETHOS: an Online Hate Speech Detection Dataset},
# #       author={Ioannis Mollas and Zoe Chrysopoulou and Stamatis Karlos and Grigorios Tsoumakas},
# #       year={2020},
# #       eprint={2006.08328},
# #       archivePrefix={arXiv},
# #       primaryClass={cs.CL}
# # }
# ###########


# class Ethos(ToxicDataSetPipe):
#     path: str = "ethos"
#     name: str = "binary"  # or "multilabel"
#     source: str = "ETHOS"

#     def __iter__(self):
#         for record in self.data:
#             example = InputRecord(
#                 source=f"{self.source}_{self.name}",
#                 labels=record["label"],
#                 text=record["text"],
#             )
#             yield example


# def callmesexistbut():
#     # dmrc-platforms/data/callmesexistbut
#     # Samory, Mattia, Indira Sen, Julian Kohne, Fabian Floeck, and Claudia Wagner. “‘Call Me Sexist, but...’: Revisiting Sexism Detection Using Psychological Scales and Adversarial Samples.” arXiv.org, April 27, 2020. https://arxiv.org/abs/2004.12764v2.

#     raise NotImplementedError()


# ###########
# # @inproceedings{gibert2018hate,
# #     title = "{Hate Speech Dataset from a White Supremacy Forum}",
# #     author = "de Gibert, Ona  and
# #       Perez, Naiara  and
# #       Garc{\'\i}a-Pablos, Aitor  and
# #       Cuadros, Montse",
# #     booktitle = "Proceedings of the 2nd Workshop on Abusive Language Online ({ALW}2)",
# #     month = oct,
# #     year = "2018",
# #     address = "Brussels, Belgium",
# #     publisher = "Association for Computational Linguistics",
# #     url = "https://www.aclweb.org/anthology/W18-5102",
# #     doi = "10.18653/v1/W18-5102",
# #     pages = "11--20",
# # }
# # https://github.com/Vicomtech/hate-speech-dataset/tree/master
# class WhiteSupremacyForum(ToxicDataSetPipe):
#     path: str = "hate_speech18"
#     source: str = "white_supremacy_forum"
#     split: str = "train"


# class BinaryHateSpeech(ToxicDataSetPipe):
#     path: str = "christinacdl/binary_hate_speech"
#     source: str = "Binary hate speech"
#     split: str = "train"

#     def __iter__(self):
#         for record in self.data:
#             example = InputRecord(
#                 source=self.source, labels=record["label"], text=record["text"]
#             )
#             yield example


# class HateSpeechOffensive(ToxicDataSetPipe):
#     path: str = "tdavidson/hate_speech_offensive"
#     source: str = "tdavidson/hate_speech_offensive"
#     split: str = "train"

#     def __iter__(self):
#         for record in self.data:
#             example = InputRecord(
#                 source=self.source,
#                 labels=[record["class"]],
#                 text=record["tweet"],
#                 count=record["count"],
#                 hate_speech_count=record["hate_speech_count"],
#                 offensive_language_count=record["offensive_language_count"],
#                 neither_count=record["neither_count"],
#             )
#             yield example


# # TODO: add https://huggingface.co/datasets/OpenSafetyLab/Salad-Data
# class SaladData(ToxicDataSetPipe):
#     path: str = "OpenSafetyLab/Salad-Data"
#     source: str = "Salad-Data"
#     split: str = "train"
