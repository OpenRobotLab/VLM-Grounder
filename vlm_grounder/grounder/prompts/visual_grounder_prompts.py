
########################################### v1
SYSTEM_PROMPTS_v1 = """Imagine you are in a room and you are aksed to find one object.

Given a series of images from a video scanning an indoor room, and a query describing a specific object in the room, you need to analyze the images to identify the room's layout and features and locate the object mentioned in the query within the images. 

You will be provided with multiple images, and the top-left corner of each image will have a number indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, I have combined multiple images ({layout}) into one image. You will also be provided with a query sentence describing the object that needs to be found, as well as a parsed version of this query, describing the target_class of the object to be found and the conditions that this object must satisfy. Please find the IDs of the images containing this object based on these conditions. Note that all the images you see contain objects of the target_class, but not necessarily the specific object you are looking for, so you need to judge whether the conditions are met.

Please return the index of the image where the object is found, and describe the object's approximate location in that image. The response should be formatted as a JSON object with the reasoning process, the index of the target image, and the position of the object in the image.

Your response should be in the following format:

```json
{{
  "reasoning": "Explain the process of how you identified the room's features and located the target object.",
  "target_image_id": "00001", # Replace with the actual image ID (only one ID) annotated on the image that contains the target object.
  "position": "Describe the position of the object in the image (e.g., top-left corner, center, bottom-right corner, etc.)"
}}
```"""

INPUT_PROMPTS_v1 = """Query: "{query}"
Target Class: {pred_target_class}
Conditions: {conditions}

Here are the {num_view_selections} images."""
#################################################


############################################### v2
SYSTEM_PROMPTS_v2 = """Imagine you are in a room and are asked to find one object.

Given a series of images from a video scanning an indoor room and a query describing a specific object in the room, you need to analyze the images to identify the room's layout and features and locate the object mentioned in the query within the images.

You will be provided with multiple images, and the top-left corner of each image will have a number indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, multiple images have been combined into one image with a layout ({layout}). You will also be provided with a query sentence describing the object that needs to be found, as well as a parsed version of this query describing the target class of the object to be found and the conditions that this object must satisfy. Please find the ID of the image containing this object based on these conditions. Note that all the images you see contain objects of the target class, but not necessarily the specific object you are looking for, so you need to judge whether the conditions are met. To get the correct answer, you may need to reason across different images.

Please return the index of the image where the object is found and describe the object's approximate location in that image. The response should be formatted as a JSON object with three keys "reasoning", "target_image_id", and "position" like this:

{{
"reasoning": "your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
"target_image_id": "00001", // Replace with the actual image ID (only one ID) annotated on the image that contains the target object.
"position": "top-left corner" // Describe the position of the object in the image (e.g., top-left corner, center, bottom-right corner, etc.)
}}"""

BBOX_SELECT_USER_PROMPTS_v1 = """Great! Here is the detailed version of your selected image. There are {num_candidate_bboxes} candidate objects shown in the image. I have annotated each object at the center with an object ID in white color text and black background. Do not mix the annotated IDs with the actual appearance of the objects. Please double check the image and the objects annotated by the IDs, and confirm whether your selected image '{image_id:05d}' really contains the target object specified by the query '{query}'. Note that you may also need to examine other images sent to you before, because the selected image may not contain all the information that is used for check the conditions. Remember, you should reason with multi-view images.
If you think the selected image is correct, we need to continue to find the target object in the image. Please select the target object by giving me the coringrect object ID.
If you think your selected image is wrong or the image does not contain the target object, tell me.
Reply using JSON format with three keys "reasoning", "image_correct_or_not", and "object_id" like this:
{{
  "reasoning": "your reasons", // Explain the justification why you think the selected image is correct and why you select the object ID, or why the image is not correct and, in this case, give hints to yourself to find the correct image next time.
  "image_correct_or_not": true, // true if you think the image is correct, false if you think the image is wrong
  "object_id": 0 // the object ID you selected when you think the image is correct; otherwise, this value is -1
}}"""

###############################################

############################################### v3
SYSTEM_PROMPTS_v3 = """You are good at finding objects specified by user queries in indoor rooms by watching the videos scanning the rooms."""

INPUT_PROMPTS_v3 = """Imagine you are in a room and are asked to find one object.

Given a series of images from a video scanning an indoor room and a query describing a specific object in the room, you need to analyze the images to locate the object mentioned in the query within the images.

You will be provided with multiple images, and the top-left corner of each image will have an ID indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, multiple images have been combined into one image with dynamic layouts. You will also be provided with a query sentence describing the object that needs to be found, as well as a parsed version of this query describing the target class of the object to be found and the conditions that this object must satisfy. Please find the ID of the image containing this object based on these conditions. Note that I have filtered the video to remove some images that do not contain objects of the target class. To locate the target object, you need to consider multiple images from different perspectives and determine which image contains the object that meets the conditions. Note, that each condition might not be judged based on just one image alone. Also, the conditions may not be accurate, so it's reasonable for the correct object not to meet all the conditions. You need to find the most possible object based on the query. If you think multiple objects are correct, simply return the one you are most confident of. If you think no objects are meeting the conditions, make a guess to avoid returning nothing. Usually the correct object is visible in multiple images, and you should return the image in which the object is most clearly observed.

Your response should be formatted as a JSON object with three keys "reasoning", "target_image_id", and "reference_image_ids" like this:

{{
  "reasoning": "your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
  "target_image_id": "00001", // Replace with the actual image ID (only one ID) annotated on the image that contains the target object.
  "reference_image_ids": ["00001", "00002", ...] // A list of IDs of images that are used to determine wether the conditions are met or not.
}}

Here is an good example: 
query: Find the black table that is surrounded by four chairs.
{{
  "reasoning": "After carefully examining all the input images, I found image 00003, 00005, and 00021 contain different tables, but only the tables in image 00003 and 00021 are black. Further, I found image 00001, image 00002, image 00003, and image 00004 show four chairs and these chairs surround the black table in image 00003. The chair in image 00005 does not meet this condition. So the correct object is the table in image 00003",
  "target_image_id": "00003",
  "reference_image_ids": ["00001", "00002", "00003", "00004"]
}}

Now start the task:
Query: "{query}"
Target Class: {pred_target_class}
Conditions: {conditions}

Here are the {num_view_selections} images for your reference."""

BBOX_SELECT_USER_PROMPTS_v3 = """Great! Here is the detailed version of your selected image. There are {num_candidate_bboxes} candidate objects shown in the image. I have annotated each object at the center with an object ID in white color text and black background. Do not mix the annotated IDs with the actual appearance of the objects. Please give me the ID of the correct target object for the query.
Reply using JSON format with two keys "reasoning" and "object_id" like this:
{{
  "reasoning": "your reasons", // Explain the justification why you select the object ID.
  "object_id": 0 // The object ID you selected. Always give one object ID from the image, which you are the most confident of, even you think the image does not contain the correct object.
}}"""


IMAGE_ID_INVALID_PROMPTS_v3 = """The image {image_id} you selected does not exist. Did you perhaps see it incorrectly? Please reconsider and select another image. Remember to reply using JSON format with the three keys "reasoning", "target_image_id", and "reference_image_ids" as required before."""

DETECTION_NOT_EXIST_PROMPTS_v3 = """The image {image_id} you selected does not seem to include any objects that fall into the category of {pred_target_class}. Please reconsider and select another image. Remember to reply using JSON format with the three keys "reasoning", "target_image_id", and "reference_image_ids" as required before."""

#############################################################

SYSTEM_PROMPTS = {
    "v1": SYSTEM_PROMPTS_v1,
    "v2": SYSTEM_PROMPTS_v2,
    "v3": SYSTEM_PROMPTS_v3,
}

INPUT_PROMPTS = {
    "v1": INPUT_PROMPTS_v1,
    "v2": INPUT_PROMPTS_v1,
    "v3": INPUT_PROMPTS_v3,
}

BBOX_SELECT_USER_PROMPTS = {
    "v1": BBOX_SELECT_USER_PROMPTS_v1,
    "v2": BBOX_SELECT_USER_PROMPTS_v1,
    "v3": BBOX_SELECT_USER_PROMPTS_v3,
}

IMAGE_ID_INVALID_PROMPTS = {
    "v1": None,
    "v2": None,
    "v3": IMAGE_ID_INVALID_PROMPTS_v3,
}

DETECTION_NOT_EXIST_PROMPTS = {
    "v1": None,
    "v2": None,
    "v3": DETECTION_NOT_EXIST_PROMPTS_v3,
}

# default version is v3