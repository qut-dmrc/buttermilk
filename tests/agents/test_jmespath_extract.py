import jmespath
import pytest
from rich import print


@pytest.fixture
def data_for_jmespath():
    return {
        "JUDGE": {
            "call_id": "0198314c-3fb2-7cca-8313-4bafbc56c49a",
            "agent_id": "JUDGE-CbQUWY",
            "outputs": {
                "conclusion": "The provided text largely adheres to the Human Rights Campaign (HRC) Foundation's guidelines for covering transgender people. It uses appropriate terminology, accurately describes the process of transition, respects the individual's name and pronouns, and focuses on the athlete as a whole person.",
                "reasons": [
                    "The text correctly uses 'transgender athlete' and 'transgender woman,' aligning with HRC's preferred usage in 'Number One: Understand what “transgender” means.'",
                    "It accurately describes Valentina Petrillo's transition as a process, stating she 'began living as a woman in 2018 before commencing hormone therapy in 2019,' which aligns with HRC's 'Number Three: Understand what “transition” means' and 'Number Four: Know that the process of transitioning isn’t always or just about surgery,' by showing both social and medical aspects.",
                    "The article consistently uses Valentina Petrillo's chosen name and 'she/her' pronouns, demonstrating respect for her identity and avoiding deadnaming, as advised in 'Number Five: Respect transgender people by using the names and pronouns they use in daily life.'",
                    "It refrains from contrasting transgender women with 'real' or 'biological' women. While it quotes opposition that implicitly makes such a distinction ('visually impaired athletes who will be beaten by Valentina Petrillo'), the article itself does not endorse this framing, consistent with HRC's 'Number Seven: Refrain from contrasting trans men and women and women with “real” or “biological” men and women.'",
                    "The story focuses on Valentina Petrillo as a complete person, highlighting her achievements as a sprinter, her visual impairment, and her personal feelings, rather than solely on her transition, which aligns with HRC's 'Number Eight: Focus on the whole person.'",
                ],
                "prediction": False,
                "uncertainty": "low",
                "preview": "∴ The provided text largely adheres to the Human Rig... | 🧹 | Uncertainty: L",
            },
            "session_id": "43a39cde-2c9d-4e6a-963c-e654fb4fd06e",
            "tracing_link": "https://wandb.ai/dmrc/bm_api-flows/r/call/0198314c-3f73-7a4f-a2f4-8c67018fe737",
            "inputs": {
                "inputs": {"query": "hrc", "prompt": "hrc"},
                "parameters": {
                    "human_in_loop": False,
                    "criteria": "hrc",
                    "save": {
                        "type": "bigquery",
                        "project_id": "prosocial-443205",
                        "dataset_id": "toxicity",
                        "table_id": "flow",
                        "dataset_name": "flow",
                        "schema_path": "flow.json",
                    },
                    "template": "judge",
                    "model": "gemini25flash",
                },
                "records": [
                    {
                        "record_id": "transgender_paralympian_skynewsau",
                        "metadata": {
                            "title": "Visually impaired sprinter Valentina Petrillo of Italy set to become first transgender athlete to compete at Paralympic Games",
                            "date": "2024-08-16T00:00:00",
                            "outlet": "Sky News Australia",
                            "fetch_source_id": "transgender_paralympian_skynewsau",
                            "fetch_timestamp_utc": "2025-07-22T08:42:13.302192+00:00",
                        },
                        "ground_truth": {
                            "reasons": [
                                "- This article discusses the inclusion of Valentina Petrillo as the first transgender athlete to compete in the Paralympic games ",
                                "- The article uses correct pronouns throughout and gender identity is relevant to the story",
                                "- The article acknowledges opposing views that are held in regards to the inclusion of transgender people in sports",
                                "- Perspectives that both support and criticise Petrillo's inclusion in the Paralympics are included",
                                "- The article does not contribute to false or harmful narratives about transgender people and overall aligns with best practice guidelines on reporting on transgender people and issues.",
                            ],
                        },
                        "content": "Italian sprinter Valentina Petrillo is set to become the first openly transgender athlete to compete at the Paralympic Games. The visually impaired competitor, 50, was selected to represent the European country in the women's 200 and 400 metre races the Games in Paris from August 28 to September 8. Petrillo last year won bronze medals in both events at the World Paris athletics championships in France\\u2019s capital. She was diagnosed with Stargardt's syndrome at 14 years old, a degenerative disease which cause...",
                    },
                ],
                "is_error": False,
            },
            "is_error": False,
            "error": [{"name": "test error dict", "value": "test error value"}],
            "object_type": "JudgeReasons",
        },
    }


def test_synth_single(data_for_jmespath):
    jmespath_expr = "[JUDGE,SYNTHESISER][].{agent_id: agent_id, result: outputs, answer_id: call_id, error: error }"
    result = jmespath.search(jmespath_expr, data_for_jmespath)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["agent_id"] == "JUDGE-CbQUWY"
    assert result[0]["result"]["conclusion"].startswith(
        "The provided text largely adheres to the Human Rights Campaign (HRC) Foundation's guidelines",
    )


if __name__ == "__main__":
    data = data_for_jmespath()
    test_synth_single(data)
    print("Test passed successfully.")
