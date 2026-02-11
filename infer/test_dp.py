from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time
import regex

import warnings
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

def exact(ans1, ans2):
    ans1 = ans1.strip().strip('.').lower()
    ans2 = ans2.strip().strip('.').lower()
    return ans1==ans2

def iou(ans1, ans2):
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'

    content_match = re.search(bbox_pattern, ans1)
    if isinstance(ans2, list):
        box2 = ans2
    else:
        sol_match = re.search(bbox_pattern, ans2)
        box2 = [float(sol_match.group(1)), float(sol_match.group(2)), float(sol_match.group(3)), float(sol_match.group(4))]
    
    if content_match:
        box1 = [float(content_match.group(1)), float(content_match.group(2)), float(content_match.group(3)), float(content_match.group(4))]
        
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    else:
        return 0

def main(bsz, data_root, model_path, num_samples, output_path, test_datasets, task_type=None):
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}, 
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)

    for ds in test_datasets:
        ds_path = os.path.join(data_root, f"{ds}.json")
        data = json.load(open(ds_path, "r"))
        random.seed(42)
        random.shuffle(data)
        data = data[:num_samples]
        
        # Split data for distributed evaluation
        per_rank_data = len(data) // world_size
        start_idx = rank * per_rank_data
        end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
        rank_data = data[start_idx:end_idx]
        
        messages = []

        for x in rank_data:
            image_path = x['image']
            message = [
                {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": f"file://{image_path}"
                    },
                    {
                        "type": "text",
                        "text": x['problem']
                    }
                ]
            }]
            messages.append(message)

        rank_outputs = [] # List to store answers for this rank
        all_outputs = []  # List to store all answers

        # Process data
        for i in tqdm(range(0, len(messages), bsz)):
            batch_messages = messages[i:i + bsz]
        
            # Preparation for inference
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            # Inference: Generation of the output
            if i_temp == 0:
                generated_ids = model.generate(
                    **inputs, 
                    use_cache=True, 
                    max_new_tokens=128, 
                    do_sample=True,          
                    temperature=1,         
                    top_p=1,
                    top_k=1
                                    
                )
            else:
                generated_ids = model.generate(
                    **inputs, 
                    use_cache=True, 
                    max_new_tokens=128, 
                    do_sample=True,          
                    temperature=i_temp,         
                    top_p=i_top_p,
                    top_k=50      
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            rank_outputs.extend(batch_output_text)

        print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

        # Gather all outputs from all ranks
        all_outputs = [None] * len(data)
        rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, rank_results)
        
        assert gathered_results[-1][-1][0] == len(data) - 1
        
        # The main process will collect all results
        if rank == 0:
            for results in gathered_results:
                for idx, output in results:
                    assert idx < len(all_outputs)
                    all_outputs[idx] = output
            assert all_outputs[-1] is not None

            final_output = []
            correct_number = 0

            for input_example, model_output in zip(data, all_outputs):
                original_output = model_output
                ground_truth = input_example['solution']
                
                if task_type=='rec':
                    model_answer = original_output
                    correct = 0
                    if model_answer is not None:
                        if iou(model_answer, ground_truth)>=0.5:
                            correct = 1
                    correct_number += correct
                else:
                    model_answer = original_output
                    correct = 0
                    if model_answer is not None:
                        if exact(model_answer, ground_truth):
                            correct = 1
                    correct_number += correct
                
                # Create a result dictionary for this example
                result = {
                    'image': input_example['image'],
                    'question': input_example['problem'],
                    'ground_truth': ground_truth,
                    'model_output': original_output,
                    'extracted_answer': model_answer,
                    'correct': correct
                }
                final_output.append(result)

            # Calculate and print accuracy
            accuracy = correct_number / len(data) * 100
            print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

            # Save results to a JSON file
            output_path_ins = output_path.format(DATASET=ds)
            output_dir = os.path.dirname(output_path_ins)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_path_ins, "w") as f:
                json.dump({
                    'accuracy': accuracy,
                    'results': final_output
                }, f, indent=2)

            print(f"Results saved to {output_path_ins}")
            print("-"*100)
            
        # Synchronize all processes
        dist.barrier()
    

if __name__ == "__main__":
    bsz=4
    data_root = "./data/" 
    num_samples = 1000000
    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"
    print(f"Process {rank} using {device}")
    
    # simulated rollout
    i_temp = 0.9
    i_top_p = 1.0
    test_datasets = ['imagenet_left_grpo'] 
    model_path=f"/path/to/qwen2_5vl_3b_cold_img/checkpoint-25/" #cold start model
    for i in range(1, 9):
        output_path="./infer/rollout_output/{DATASET}_qwen2_5vl_3b_cold_img_" + str(i) + ".json"
        main(bsz, data_root, model_path, num_samples, output_path, test_datasets, task_type='math')
    
    #evaluation
    i_temp = 0
    i_top_p = 1.0
    test_datasets = ['imagenet_val_processed',  'imagenet-r_val_processed', 'imagenet-a_val_processed']
    model_path=f"/path/to/qwen2_5vl_3b_cold_img_sft/checkpoint-600/" # sft model
    output_path="./infer/eval_output/{DATASET}_qwen2_5vl_7b_cold_img_sft.json/"
    main(bsz, data_root, model_path, num_samples, output_path, test_datasets)
