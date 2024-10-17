import argparse
import json
import os

import mmengine
import pandas as pd
from mmengine.utils.dl_utils import TimeCounter
from tqdm import tqdm

from vlm_grounder.utils import CategoryJudger, OpenAIGPT

DEFAULT_OPENAIGPT_CONFIG = {"temperature": 1, "top_p": 1, "max_tokens": 4095}

query_analysis_SYSTEM_PROMPT = {
    "v1": """You are working on a 3D visual grounding task, which involves receiving a query that specifies a particular object by describing its attributes and grounding conditions to uniquely identify the object. Here, attributes refer to the inherent properties of the object, such as category, color, appearance, function, etc. Grounding conditions refer to considerations of other objects or other conditions in the scene, such as location, relative position to other objects, etc.
Now, I need you to first parse this query, return the category of the object to be found, and list each of the object's attributes and grounding conditions. Each attribute and condition should be returned individually. Sometimes the object's category is not explicitly specified, and you need to deduce it through reasoning.
Your response should be formatted using JSON contained in ```json```.

Here are some examples:

Input:
Query: Printer on left, shelving on right, you want the back option of the two boxes

Output:
```
{
"target_class": "box",
"attributes": [], # no attributes
"conditions": ["a printer is on its left", "a shelving is on its right", "the back option"]
}
```

Input:
Query: Find the box with the longest set of bookcases, then choose the smaller one in the corner next to it on the left.

Output:
```
{
"target_class": "box",
"attributes": [], # no attributes
"conditions": ["it's in the corner", "it's the smaller one", "it's next to and on the left of a box with the longest set of bookcases"]
}

Input:
Query: this is a armchair. its white with patterns on it. its closest to the window in the room.

Output:
```
{
"target_class": "armchair",
"attributes": ["it's white", "it has patterns on it"],
"conditions": ["it's closest to the window in the room"]
}
```
Ensure your response adheres strictly to this JSON format, as it will be directly parsed and used.""",
    "v2": """You are working on a 3D visual grounding task, which involves receiving a query that specifies a particular object by describing its attributes and grounding conditions to uniquely identify the object. Here, attributes refer to the inherent properties of the object, such as category, color, appearance, function, etc. Grounding conditions refer to considerations of other objects or other conditions in the scene, such as location, relative position to other objects, etc.
Now, I need you to first parse this query, return the category of the object to be found, and list each of the object's attributes and grounding conditions. Each attribute and condition should be returned individually. Sometimes the object's category is not explicitly specified, and you need to deduce it through reasoning. If you cannot deduce after reasoning, you can use 'unknown' for the category.
Your response should be formatted using JSON contained in ```json```.

Here are some examples:

Input:
Query: this is a brown cabinet. it is to the right of a picture.

Output:
```
{
"target_class": "cabinet",
"attributes": ["it's brown"],
"conditions": ["it's to the right of a picture"]
}
```

Input:
Query: it is a wooden computer desk. the desk is in the sleeping area, across from the living room. the desk is in the corner of the room, between the nightstand and where the shelf and window are.

Output:
```
{
"target_class": "desk",
"attributes": ["it's a wooden computer desk"],
"conditions": ["it's in the sleeping area", "it's across from the living room", "it's in the corner of the room", "it's between the nightstand and where the shelf and window are"]
}
```

Input:
Query: In the room is a set of desks along a wall with windows totaling 4 desks. Opposite this wall is another wall with a door and two desks. The desk of interest is the closest desk to the door. This desk has nothing on it, no monitor, etc.

Output:
```
{
"target_class": "desk",
"attributes": [],  # no attributes
"conditions": ["it's near the wall with a door and two desks, opposite a wall with windows totaling four desks", "it's closest to the door on the wall", "it has nothing on it, no monitor, etc"] # need some reasoning to write the conditions
}
```
Ensure your response adheres strictly to this JSON format, as it will be directly parsed and used.""",
}


class QueryAnalysis:
    def __init__(
        self,
        model,
        output_dir,
        scene_info_file=None,
        vg_file=None,
        use_global_cache=True,
        prompt_version=1,
        result_prefix="",
    ) -> None:
        self.prompt_version = prompt_version
        self.result_prefix = result_prefix

        self.openai_gpt_type = model
        self.model = OpenAIGPT(model=model, **DEFAULT_OPENAIGPT_CONFIG)

        self.use_global_cache = (
            use_global_cache  # * cannot use when scene_info are not GT
        )

        # * load the the scene_info_file and vg_file if they are not none and are str
        # * if str but not exists, raise error, else not str, directly assign as they are the opened contents
        if isinstance(scene_info_file, str):
            if not mmengine.exists(scene_info_file):
                raise FileNotFoundError(f"File {scene_info_file} not found.")
            self.scene_info_file = mmengine.load(scene_info_file)
        else:
            self.scene_info_file = scene_info_file
        if isinstance(vg_file, str):
            if not mmengine.exists(vg_file):
                raise FileNotFoundError(f"File {vg_file} not found.")
            self.vg_file = mmengine.load(vg_file)
        else:
            self.vg_file = vg_file

        self.output_dir = f"{output_dir}/intermediate_results"
        # * remember to use global cache, make sure the scene_info is correct
        self.global_cache_output_file_path = (
            f"outputs/global_cache/query_analysis_v{prompt_version}/global_cache.json"
        )

        mmengine.mkdir_or_exist(os.path.dirname(self.global_cache_output_file_path))
        mmengine.mkdir_or_exist(self.output_dir)

    def chat_complete(self, prompt):
        messages = [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["input"]},
        ]
        gpt_response = self.model.safe_chat_complete(
            messages, response_format={"type": "json_object"}
        )

        return gpt_response

    def format_input_prompt_query_analysis(self, **kwargs):
        query = kwargs["query"]
        return query

    def format_prompt(self, **kwargs):
        """
        Return:
            dict with keys {"system_prompt", "input"}
        """
        format_prompt_func = eval(f"self.format_input_prompt_query_analysis")
        input_prompt = format_prompt_func(**kwargs)
        system_prompt = eval(f"query_analysis_SYSTEM_PROMPT")[
            f"v{self.prompt_version}"
        ]

        return {"system_prompt": system_prompt, "input": input_prompt}

    @TimeCounter(tag="QueryAnalysis", log_interval=50)
    def invoke(self, scene_id, query):
        # * look up global cache
        # * a global cache is a dict with key (scene_id, query, model) as key
        # * load the result if exists
        if self.use_global_cache:
            if mmengine.exists(self.global_cache_output_file_path):
                self.global_cache = mmengine.load(self.global_cache_output_file_path)
                self.global_cache = {
                    eval(key): value for key, value in self.global_cache.items()
                }
            else:
                self.global_cache = {}

        output_path = os.path.join(self.output_dir, scene_id)
        mmengine.mkdir_or_exist(output_path)
        output_file_path = f"{output_path}/{self.result_prefix}_results.json"
        # * if this file exits, then should load the file otherwise, use an empty list
        if mmengine.exists(output_file_path):
            results = mmengine.load(output_file_path)
        else:
            results = []

        if (
            self.use_global_cache
            and (scene_id, query, self.openai_gpt_type) in self.global_cache
        ):
            result = self.global_cache[(scene_id, query, self.openai_gpt_type)]
            result["from_cache"] = True
            results.append(result)
        else:
            # * use GPT to get new_result
            # * while loop, sometimes the response may have wrong format
            prompt = self.format_prompt(scene_id=scene_id, query=query)

            retry_time = 0
            while True:
                gpt_response = self.chat_complete(prompt)
                parse_result = self.parse_gpt_response(gpt_response["content"])
                # * save one result
                flag = parse_result["flag"]
                result = {
                    "scene_id": scene_id,
                    "query": query,
                    "flag": flag,
                    "parse_status": parse_result["parse_status"],
                    "task_result": parse_result["task_result"],
                    "prompt": prompt,
                    "gpt_response": gpt_response,
                    "retry_time": retry_time,
                    "from_cache": False,
                }
                results.append(result)
                if flag:
                    break
                else:
                    retry_time += 1
                    print(f"Error: {parse_result}. Retrying {retry_time}.")
                    print(f"GPT response: {gpt_response}")
                # * if exceed try time 20, then exit and raise error
                if retry_time > 20:
                    raise ValueError(f"Error: {parse_result}. Retry time exceeds 20.")

            if self.use_global_cache:
                # * update and save the global cache
                self.global_cache[(scene_id, query, self.openai_gpt_type)] = result
                # * before dump, need to convert to str

                mmengine.dump(
                    {str(key): value for key, value in self.global_cache.items()},
                    self.global_cache_output_file_path,
                    indent=2,
                )

        # * save the current results
        mmengine.dump(results, output_file_path, indent=2)

        return result["task_result"]

    def parse_gpt_response(self, gpt_content):
        parse_result = {"flag": False, "parse_status": "", "task_result": None}
        try:
            gpt_content_json = json.loads(gpt_content)
            task_result = eval(f"self.get_query_analysis_result")(gpt_content_json)

            # Update the parse result on success
            parse_result.update(
                {"flag": True, "parse_status": "Success", "task_result": task_result}
            )

        except Exception as e:
            parse_result["parse_status"] = str(e)

        return parse_result

    def get_query_analysis_result(self, response_json):
        if "target_class" not in response_json:
            raise ValueError("JSON parsing error: 'target_class' is missing")

        target_class = response_json["target_class"]
        attributes = response_json.get("attributes", [])
        conditions = response_json.get("conditions", [])

        if not isinstance(attributes, list) or not isinstance(conditions, list):
            raise ValueError(
                "Data extraction error: 'attributes' or 'conditions' is not a list"
            )

        query_result = {
            "pred_target_class": target_class,
            "attributes": attributes,
            "conditions": conditions,
        }
        return query_result


def calculate_analysis_accuracy(data):
    category_judger = CategoryJudger()
    data["pred_target_class"] = data["pred_target_class"].str.replace(" ", "_")

    matches = 0
    mismatches = []

    for index, row in data.iterrows():
        target_class_processed = row["pred_target_class"]
        instance_type = row["instance_type"]
        query = row["utterance"]  # 假设原始查询存储在 'utterance' 列

        if category_judger.is_same_category(target_class_processed, instance_type):
            matches += 1
        else:
            mismatches.append(
                (index, row["scan_id"], target_class_processed, instance_type, query)
            )

    accuracy = matches / len(data) if len(data) else 0

    print(f"Accuracy of category matching: {accuracy:.2f}")
    if mismatches:
        print("Mismatched items:")
        for mismatch in mismatches:
            print(
                f"Index: {mismatch[0]}, Scan ID: {mismatch[1]}, Target Class: {mismatch[2]}, Instance Type: {mismatch[3]}, Query: {mismatch[4]}"
            )


if __name__ == "__main__":
    # vg_file_path = "data/scannet/grounding/referit3d/scanrefer_sampled_50_relations.csv"
    # result_prefix = "scanrefer_test50" # To store information of each openai request

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vg_file",
        type=str,
        default="data/scannet/grounding/referit3d/scanrefer_sampled_50_relations.csv",
    )  # * no used
    parser.add_argument("--result_prefix", type=str, default="")
    args = parser.parse_args()
    vg_file_path = args.vg_file
    result_prefix = args.result_prefix
    print(vg_file_path)

    model = "gpt-4o-2024-05-13"
    prompt_version = 2
    output_dir = f"query_analysis_v{prompt_version}/"
    vg_file = pd.read_csv(vg_file_path)

    finder = QueryAnalysis(
        model,
        scene_info_file=None,
        vg_file=None,
        output_dir=output_dir,
        result_prefix=result_prefix,
        prompt_version=prompt_version,
    )

    target_classes = []
    attributes = []
    conditions = []

    for index, row in tqdm(vg_file.iterrows()):
        scene_id = row["scan_id"]
        query = row["utterance"]

        task_result = finder.invoke(scene_id, query)

        if task_result is None:
            print(f"Error: No result for scene_id {scene_id}, query '{query[0:60]}'")
            # use some default values
            task_result = {
                "pred_target_class": "GPT Fail",
                "attributes": [],
                "conditions": [],
            }

        target_classes.append(task_result["pred_target_class"])
        attributes.append(task_result["attributes"])
        conditions.append(task_result["conditions"])

    vg_file["pred_target_class"] = target_classes
    vg_file["attributes"] = attributes
    vg_file["conditions"] = conditions

    mmengine.mkdir_or_exist("outputs/query_analysis")
    new_vg_file_path = f"outputs/query_analysis/prompt_v{prompt_version}_updated_{os.path.basename(vg_file_path)}"
    vg_file.to_csv(new_vg_file_path, index=False)

    # Calculate accuracy
    calculate_analysis_accuracy(vg_file)
