
from promptflow import tool
from buttermilk.utils.log import logger

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need

@tool
def score(moderated_results: dict, synth_results: dict, groundtruth: dict, record_id: str) -> dict:
    num_correct = 0
    num_results = 0
    overall_correct = 0
    overall_results = 0

    expected = groundtruth['answer']

    scored_results = dict(moderated={}, synthesised={}, record_id=record_id)

    for name, result in moderated_results.items():
        try:
            pred = result.get('predicted')
            correct = int(pred) == int(expected)
            scored_results['moderated'][name] = {}
            scored_results['moderated'][name]['correct'] = correct
            if correct:
                num_correct += 1
            num_results += 1
        except Exception as e:
            logger.warning(f"Scorer hit error processing {name} with value {result}: {e} {e.args=}")
            pass

    for name, result in synth_results.items():
        try:
            scored_results['synthesised'][name] = {}
            scored_results['synthesised'][name]['corrent_inputs'] = num_correct
            scored_results['synthesised'][name]['total_inputs'] = num_results
            pred = result.get('predicted')
            correct = int(pred) == int(expected)
            scored_results['synthesised'][name]['correct'] = correct
            if correct:
                overall_correct += 1
                overall_results += 1
        except Exception as e:
            logger.warning(f"Scorer hit error processing {name} with value {result}: {e} {e.args=}")
            pass

        scored_results["overall"] = dict(correct=overall_correct, total=overall_results)
        scored_results["expected"] = expected

    return scored_results
