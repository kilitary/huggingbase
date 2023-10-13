import os
import json

import datasets


def parse_json(x):
    return json.loads(x)


_DESCRIPTION = ""
_URL = "ru_instruct_gpt4.jsonl"


class RuInstructGPT4Dataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description=""),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features(
            {
                "instruction": datasets.Value("string"),
                "input"      : datasets.Value("string"),
                "output"     : datasets.Value("string"),
                "full_output": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features
        )

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"path": downloaded_file}),
        ]

    def _generate_examples(self, path):
        with open(path, "r") as f:
            for id_, line in enumerate(f):
                yield id_, parse_json(line)
