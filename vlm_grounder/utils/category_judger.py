import os

import mmengine
from mmengine.utils.dl_utils import TimeCounter
from nltk.corpus import wordnet

from .my_openai import OpenAIGPT


class CategoryJudger():
    def __init__(self, openai_model="gpt-3.5-turbo-0125", cache_dir="outputs/global_cache/category_judger"):
        self.openai_cost = 0.0
        self.cache_dir = f"{cache_dir}/{openai_model}"
        self.cache_file = os.path.join(self.cache_dir, "category_cache.json")
        self.cache = self.load_cache()
        self.openai_model = openai_model

    def load_cache(self):
        mmengine.mkdir_or_exist(self.cache_dir)
        if os.path.exists(self.cache_file):
            return mmengine.load(self.cache_file)
        return {}

    def save_cache(self):
        mmengine.dump(self.cache, self.cache_file, indent=2)

    def is_same_category(self, n1: str, n2: str) -> bool:
        """
        Use string match, wordnet and GPT to determine whether n1 and n2 are in the same category.
        """
        # if n1 or n2 is "unknown", then directly return False
        if n1 == "unknown" or n2 == "unknown":
            return False
        key = ", ".join(sorted([n1, n2]))
        if key in self.cache:
            return self.cache[key]
        
        result = self.is_same_category_strmatch(n1, n2) or \
                self.is_same_category_wordnet(n1, n2) or \
                self.is_same_category_openai(n1, n2)

        self.cache[key] = result
        self.save_cache()
        return result

    # * TODO: Implement a batch version

    # @TimeCounter(tag="WordNetSameCategory")
    def is_same_category_wordnet(self, noun1: str, noun2: str) -> bool:
        synsets1 = wordnet.synsets(noun1, pos=wordnet.NOUN)
        synsets2 = wordnet.synsets(noun2, pos=wordnet.NOUN)
        # Check if two nouns have the same synonym set
        for synset1 in synsets1:
            for synset2 in synsets2:
                if synset1 == synset2:
                    return True
        return False

    # @TimeCounter(tag="StrMatchSameCategory")
    # * if remove white space and upper/lower letter exactly the same
    def is_same_category_strmatch(self, noun1: str, noun2: str) -> bool:
        return noun1.strip().lower() == noun2.strip().lower()

    # @TimeCounter(tag="OpenAISameCategory")
    def is_same_category_openai(self, noun1: str, noun2: str) -> bool:
        if isinstance(self.openai_model, str):
            # lazy init, init only when used
            self.openai_model = OpenAIGPT(model=self.openai_model, max_tokens=4)

        global total_cost
        system_prompt = """Given two noun phrases, you need to determine if these two noun phrases could possibly refer to the same type of object in real-life scenarios. If one contains the other, then they are considered the same object. If they are the same type of object, you should directly reply with True or False, without anything else. Here are some examples:
    Input: desk, table
    Output: True

    Input: kitchen_cabinets, cabinet
    Output: True

    Input: office chair, chair
    Output: True

    Input: dinning table, chair
    Output: False

    Input: man, woman
    Output: False"""
        input_prompt = f"{noun1}, {noun2}"
        gpt_result = self.openai_model.single_round_chat(system_prompt, input_prompt)
        cls_result = gpt_result['content']
        cost = gpt_result['cost']
        self.openai_cost += cost
        print(f"cost: {cost}")
        if cls_result.lower() == "true":
            return True
        elif cls_result.lower() == "false":
            return False
        else:
            print("OpenAI GPT is_same_category error!")
