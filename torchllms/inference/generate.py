import argparse
import json
from pathlib import Path

import jsonlines

from torchllms.inference import providers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path of input .jsonl file. Expects a field 'messages' with a list of messages, or a field 'prompt' with a completion prompt.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path of output .jsonl file. Copies content of input file and extends the 'messages' field with assistant responses or fills the 'completion' field with completion.",
    )
    parser.add_argument("--provider", type=str, required=True, help="Model provider")
    parser.add_argument(
        "--provider_kwargs",
        type=json.loads,
        default={},
        help="JSON-encoded kwargs for provider constructor",
    )
    parser.add_argument(
        "--generate_kwargs",
        type=json.loads,
        default={},
        help="JSON-encoded kwargs for provider.generate call",
    )
    args = parser.parse_args()

    provider = providers.PROVIDERS[args.provider](**args.provider_kwargs)


    with jsonlines.open(args.input_path) as reader:
        inputs = list(reader)

    if "messages" in inputs[0]:
        conversations = [conversation["messages"] for conversation in inputs]
        conversations = provider.generate(
            conversations=conversations, **args.generate_kwargs
        )
        for inp, conv in zip(inputs, conversations):
            inp["messages"] = conv
    else:
        prompts = [conversation["prompt"] for conversation in inputs]
        completions = provider.generate(prompts=prompts, **args.generate_kwargs)
        for inp, cmpl in zip(inputs, completions):
            inp["completion"] = cmpl
    

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(args.output_path, "w") as writer:
        writer.write_all(inputs)


if __name__ == "__main__":
    main()
