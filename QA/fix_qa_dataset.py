import json
from collections import defaultdict

SINGLE_HOP_INPUT = "QA/single_hop_qa.json"
MULTI_HOP_INPUT = "QA/multi_hop_qa.json"
CHAINS_INPUT = "QA/multi_hop_chains.json"

SINGLE_HOP_OUTPUT = "QA/single_hop_qa_fixed.json"
MULTI_HOP_OUTPUT = "QA/multi_hop_qa_fixed.json"


def fix_single_hop():
    with open(SINGLE_HOP_INPUT, 'r') as f:
        data = json.load(f)

    # group by question text
    groups = defaultdict(lambda: {
        "question": None,
        "answer_nodes": [],
        "relation": None,
        "direction": None,
        "subject_id": None,
        "subject_name": None,
        "hop": 1
    })

    for item in data:
        key = item['question']
        group = groups[key]
        group['question'] = item['question']
        group['relation'] = item['relation']
        group['direction'] = item['direction']
        group['subject_id'] = item['subject_id']
        group['subject_name'] = item['subject_name']
        group['hop'] = item['hop']

        # add answer if not already in list
        answer = {"id": item['answer_node_id'], "name": item['answer_name']}
        if answer not in group['answer_nodes']:
            group['answer_nodes'].append(answer)

    result = list(groups.values())

    with open(SINGLE_HOP_OUTPUT, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Single-hop: {len(data)} pairs → {len(result)} unique questions")
    return result


def fix_multi_hop():
    with open(CHAINS_INPUT, 'r') as f:
        chains = json.load(f)

    with open(MULTI_HOP_INPUT, 'r') as f:
        qa_pairs = json.load(f)

    # build a lookup from chain index to GPT generated question
    # multi_hop_qa and chains are in the same order
    chain_to_question = {}
    for i, qa in enumerate(qa_pairs):
        if i < len(chains):
            key = (
                chains[i]['start_node_id'],
                tuple(chains[i]['relations']),
                chains[i]['hop']
            )
            if key not in chain_to_question:
                chain_to_question[key] = qa.get('question', '') if isinstance(qa, dict) else qa

    # group chains by (start_node_id, relations, hop)
    groups = defaultdict(lambda: {
        "question": None,
        "answer_nodes": [],
        "start_node_id": None,
        "start_node_name": None,
        "relations": None,
        "hop": None,
        "chain_text": None
    })

    for chain in chains:
        key = (
            chain['start_node_id'],
            tuple(chain['relations']),
            chain['hop']
        )

        group = groups[key]
        group['start_node_id'] = chain['start_node_id']
        group['start_node_name'] = chain['start_node_name']
        group['relations'] = chain['relations']
        group['hop'] = chain['hop']
        group['chain_text'] = chain['chain_text']

        # use GPT question if available
        if group['question'] is None:
            group['question'] = chain_to_question.get(key, chain['chain_text'])

        # add answer if not already in list
        answer = {"id": chain['answer_node_id'], "name": chain['answer_name']}
        if answer not in group['answer_nodes']:
            group['answer_nodes'].append(answer)

    result = list(groups.values())

    with open(MULTI_HOP_OUTPUT, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Multi-hop: {len(chains)} chains → {len(result)} unique questions")
    return result


if __name__ == "__main__":
    print("Fixing single-hop QA...")
    single = fix_single_hop()
    print(f"Sample: {single[0]}")
    print()
    print("Fixing multi-hop QA...")
    multi = fix_multi_hop()
    print(f"Sample: {multi[0]}")
    print()
    print("Done! Fixed files saved.")