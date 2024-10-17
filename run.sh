#!/usr/bin/zsh
source ~/.zshrc

# initial visual grounding
# TO CHANGE
VG_FILE=outputs/query_analysis/*_relations_with_images_selected_diffconf_and_pkl.csv

# TO CHANGE
DET_INFO=outputs/image_instance_detector/*/chunk*/detection.pkl

# TO CHANGE
MATCH_INFO=data/scannet/scannet_match_data/*.pkl

DATE=2024-06-21
EXP_NAME=test 

GPT_TYPE=gpt-4o-2024-05-13
PROMPT_VERSION=3

python ./sceneagent/agents/visual_grouding_video.py \
  --from_scratch \
  --do_ensemble \
  --post_process_component \
  --post_process_erosion \
  --use_sam_huge \
  --use_bbox_prompt \
  --vg_file_path ${VG_FILE} \
  --exp_name ${DATE}_${EXP_NAME} \
  --prompt_version ${PROMPT_VERSION} \
  --openaigpt_type ${GPT_TYPE} \
  --skip_bbox_selection_when1 \
  --det_info_path ${DET_INFO} \
  --matching_info_path ${MATCH_INFO} \
  --use_new_detections \
  --dynamic_stitching \
  --kernel_size 7 \
  --online_detector yolo \
