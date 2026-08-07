import time
from swift.llm import (
    get_model_tokenizer, get_template, ModelType, load_dataset, EncodePreprocessor
)
from swift.utils import seed_everything
from swift.tuners import Swift
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union,Literal
from swift.llm import RequestConfig,InferRequest,TemplateInputs
from swift.llm import InferClient, InferRequest
import ray 

import  torch
import torch.nn as nn
from loguru import logger
import copy
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing_vlm import DataProcessor,trojectory_example_prompt,denormalize_with_params
from video_tool import images_get_from_video, video_trajectory
import tqdm
import math
from collections import defaultdict
import heapq
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import random
import numpy as np

def clip_one(image):
    width, height = image.size

    if width > height:
        left = (width - height) // 2
        right = left + height
        top = 0
        bottom = height
    else:
        top = (height - width) // 2
        bottom = top + width
        left = 0
        right = width
    square_image = image.crop((left, top, right, bottom))
    image=square_image
    return image

def to_device(data: Any, device: Union[str, torch.device, int]) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device)
    else:
        return data


class GAC_model():
    def __init__(self,tag='critic'):
        self.tag=tag
        self.temperature=1
        self.max_tokens=10240
        self.do_sample=True
        self.top_logprobs=10
        self.logprobs=True
        self.top_k=None
        self.system_prompt=None
        self.action_data={}
        self.critic_data={}
        self.sft_dataset=None
        self.dataclient=DataProcessor()
        self.dataclient.prompt_templete['v3']="Image-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image. + score means the second image is closer, - score means the first image is closer\nResponse the relative progressing of target task follow <score>. The target task is: <task> {} </task> <score>"
        self.dataclient.prompt_templete['v3_think']="0% <image>\nThis image is the trajectory beginning of the following two images\nImage-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image. + score means the second image is closer, - score means the first image is closer\nResponse the relative progressing of target task follow <score>. The target task is: <task> {} </task> <score>"


    def _songling_process(self,action):
        action=copy.deepcopy(action)
        for i in range(len(action)):
            if i in [0, 1, 2]:
                action[i]=action[i]//1000
            elif i in [3,4,5]:
                action[i]=action[i]//1000
            elif i ==6:
                pass
            else:
                pass
        return action

    def format_state(self,state,gripper_format=False):
        state=self._songling_process(state)
        if gripper_format:
            state_open=round((state[6]//10000)/7.0,1)
            state[6]=state_open
        else:
            state[6]=state[6]//1000
        return state

    def get_score_prompt(self,task,trajectory_len=0,think=False):
        "two or len+3 image"
        if trajectory_len>0:
            trajectory_prompt=trojectory_example_prompt(list(range(trajectory_len)),task=task)
            full_prompt=trajectory_prompt+self.dataclient.prompt_templete['v3_think'].format(task)
        else:
            full_prompt=self.dataclient.prompt_templete['v3'].format(task)
        if think:
            full_prompt+=self.dataclient.prompt_templete['think']
        return full_prompt
    def get_done_prompt(self,task):
        "one image"
        return self.dataclient.prompt_templete["task_done"].format(task)
    
    def get_in_context_done_prompt(self,task=None):
        "two image"
        if task:
            return self.dataclient.prompt_templete["context_task_done"].format(task,task)
        else:
            return self.dataclient.prompt_templete["image_done"]
    
    def get_task_prompt(self):
        "two image"
        return self.dataclient.prompt_templete["task_vqa"]
    def get_action_inverse_prompt(self):
        "two image"
        return self.dataclient.prompt_templete["action_inverse"]
    
    def get_action_score_prompt(self,task,state,action):
        "one image"
        format_fuction=self.dataclient.action_format['songling']
        action_score_prompt=self.dataclient.prompt_templete['task_action_score'].format(format_fuction(state),format_fuction(action),task)
        return action_score_prompt

    def get_action_prompt(self,task,view_num=1,position_output=False,simple=False,state='type',output_num=1,think=False):
        if simple:
            format_fuction=self.dataclient.action_format_simple['songling']
            action_key='task_action_simple'
            if position_output:
                action_key='task_position_simple'
        else:
            format_fuction=self.dataclient.action_format['songling']
            action_key='task_action'
            if position_output:
                action_key='task_position'
        "view_num(max 3) image"
        image_prompt=self.dataclient.image_prompt_templete[view_num]
        full_prompt=image_prompt+self.dataclient.prompt_templete[action_key].format(format_fuction(state,state=position_output),task)
        if output_num>1:
            full_prompt+=f'*{output_num}'
        if think:
            full_prompt+=self.dataclient.prompt_templete['think']
        return full_prompt
    
    def get_fast_action_prompt(self,task,view_num=1,position_output=False,simple=False,state='type',output_num=1,think=False):
        full_prompt = self.get_action_prompt(task,view_num,position_output,simple,state,output_num,think)
        full_prompt+='<chunks>'
        return full_prompt
    def set_config(self):
        self.request_config = RequestConfig(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )
        self.model.generation_config.max_new_tokens = self.max_tokens
        self.model.generation_config.do_sample=self.do_sample
        self.model.generation_config.temperature=self.temperature

    def _get_internvl2_per_token_logps(self, model, inputs):
        from trl.trainer.utils import selective_log_softmax
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        inputs = {
            k: v
            for k, v in inputs.items() if k not in
            ['logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps']
        }
        _, inputs = self.template.pre_forward_hook(self.model, None, inputs)
        logits = model(**inputs).logits
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        logits = logits / self.temperature
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)
     
    def init_model(self,model_path,model_type='internvl2',device_map:str = 'auto',torch_dtype=torch.bfloat16,adapter: str = None):
        """
        Args:
            device_map: ['auto', 'cuda:0',...]
        """
        template_type = model_type
        print(f'template_type: {template_type}')
        self.model, tokenizer = get_model_tokenizer(model_id_or_path=model_path,
                                            model_type=model_type,
                                            torch_dtype=torch_dtype,
                                            device_map=device_map,
                                            attn_impl = 'flash_attn')
        self.template = get_template(template_type, tokenizer)
        if adapter:
            self.model = Swift.from_pretrained(self.model, adapter, adapter_name=None)

        from swift.llm import PtEngine
        from swift.plugin import InferStats
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)

        self.infer_stats = InferStats()
        seed_everything(42)
        logger.success("model initialized successfully")
   
    def chat(self,infer_requests):
        start_t=time.time()
        response_list = self.engine.infer(
        infer_requests, template=self.template, request_config=self.request_config, metrics=[self.infer_stats])
        end_t=time.time()
        infer_time=end_t-start_t
        return response_list,infer_time
    
    def results_format(self,response_list,infer_requests,rich=False):
        infer_requests=copy.deepcopy(infer_requests)
        answers=[]
        for i in range(len(response_list)):
            if rich:
                rich_answer=''
                for one in response_list[i].choices[0].logprobs['content'][:-1]:
                    if one['token'].isdigit():
                        temp_num=0
                        temp_weight=0
                        top_prob=math.e**one['top_logprobs'][0]['logprob']
                        for one_tops in one['top_logprobs']:
                            top_num=one_tops['token']
                            if top_num.isdigit():
                                prob=math.e**one_tops['logprob']
                                if prob>top_prob*0.1:
                                    temp_num+=float(top_num)*prob
                                    temp_weight+=prob
                        rich_answer+="{:.1f}".format(temp_num/temp_weight)
                    else:
                        rich_answer+=one['token']
                answers.append(rich_answer)
            else:
                answers.append(response_list[i].choices[0].message.content)
            infer_requests[i].messages.append({'role': 'assistant', 'content': response_list[i].choices[0].message.content})
        return answers,infer_requests
    

    def fast_results_format(self,response_list,infer_requests,tokenizer,time_horizon=10,action_dim=7):
        infer_requests=copy.deepcopy(infer_requests)
        answers=[]
        for i in range(len(response_list)):
            temp=[]
            text=[]
            for one in response_list[i].choices[0].logprobs['content'][:-1]:
                ids=tokenizer.encode(one['token'],add_special_tokens=False)
                if 92537-ids[0]<=2048:
                    temp.extend(ids)
            temp=[[92537-one for one in temp]]
            fast_chunk=tokenizer.fasttokenizer.decode(temp,time_horizon=time_horizon,action_dim=action_dim)
            infer_requests[i].messages.append({'role': 'assistant', 'content': response_list[i].choices[0].message.content})
        return fast_chunk,infer_requests
    
    def denormalize_fastchunk(self,fast_chunk,params):
        return denormalize_with_params(fast_chunk,params)

    def set_system_prompt(self,system_prompt=None):
        if system_prompt is not None:
            self.system_prompt=system_prompt
        elif system_prompt=='default':
            self.system_prompt=None
        else:
            self.system_prompt='You are a visual-language assistant designed to interpret spatial and task-related information from images and text. Provide precise, context-aware responses and actionable guidance to assist in achieving task objectives.'
    
    def _process_image_to_pil(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        将各种格式的图像输入转换为448x448的PIL.Image
        
        Args:
            image_input: 图像URL、文件路径、base64编码字符串或PIL.Image对象
        
        Returns:
            PIL.Image: 调整为448x448尺寸的PIL图像
        """
        pil_image = None
        
        if isinstance(image_input, Image.Image):
            pil_image = image_input
        elif isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input)
                pil_image = Image.open(BytesIO(response.content))
            elif image_input.startswith('data:image'):
                header, encoded = image_input.split(',', 1)
                image_data = base64.b64decode(encoded)
                pil_image = Image.open(BytesIO(image_data))
            elif len(image_input) > 100 and not '/' in image_input and not '\\' in image_input:
                try:
                    image_data = base64.b64decode(image_input)
                    pil_image = Image.open(BytesIO(image_data))
                except:
                    pil_image = Image.open(image_input)
            else:
                pil_image = Image.open(image_input)
        else:
            raise ValueError(f"不支持的图像输入类型: {type(image_input)}")
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        pil_image = pil_image.resize((448, 448), Image.Resampling.LANCZOS)
        
        return pil_image
    
    def get_infer_requests(self,prompt,images:List[Union[str, Image.Image]] = None):
        """
        Args:
            prompt: 提示文本
            images: 可以是图像URL、文件路径、base64编码字符串或PIL.Image对象
        """
        if type(prompt)==str:
            prompt=[prompt]
            images=[images]
        infer_requests=[]
        for i in range(len(prompt)):
            one_input=[]
            if self.system_prompt:
                one_input.append(
                    {
                        'role': 'system',
                        'content': self.system_prompt
                    }
                )
            one_input.append(
                {
                    'role': 'user',
                    'content': prompt[i]
                }
            )
            processed_images = None
            if images[i] is not None:
                if isinstance(images[i], list):
                    processed_images = [self._process_image_to_pil(img) for img in images[i] if img is not None]
                else:
                    processed_images = [self._process_image_to_pil(images[i])]
            
            infer_requests.append(TemplateInputs(
                messages=one_input,
                images=processed_images,
            ))
            # infer_requests.append(TemplateInputs(
        return infer_requests
    
    def get_logprobs(self,infer_requests,return_mask=False,digit=False):
        old_mode=self.template.mode
        self.template.set_mode('train')
        mini_batch_encoded_inputs = [self.template.encode(infer_request) for infer_request in infer_requests]
        mini_batch_encoded_inputs = to_device(
            self.template.data_collator(mini_batch_encoded_inputs), self.model.device)
        labels = mini_batch_encoded_inputs.pop('labels')
        logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
        mini_batch_encoded_inputs['logits_to_keep'] = logits_to_keep
        mini_batch_encoded_inputs['completion_mask'] = labels[:, -logits_to_keep:] != -100
        per_token_logps = self._get_internvl2_per_token_logps(self.model, mini_batch_encoded_inputs)
        self.template.set_mode(old_mode)
        if return_mask:
            if digit:
                tokenizer = self.template.tokenizer
                
                digit_completion_mask = torch.zeros_like(labels[:, -logits_to_keep:], dtype=torch.bool)
                for i in range(labels.shape[0]):
                    for j in range(labels.shape[1] - logits_to_keep, labels.shape[1]):
                        if labels[i, j] != -100:
                            token_id = labels[i, j].item()
                            token_str = tokenizer.decode([token_id]).strip()
                            if re.search(r'[0-9+\-]', token_str):
                                digit_completion_mask[i, j - (labels.shape[1] - logits_to_keep)] = True
                return per_token_logps, mini_batch_encoded_inputs['completion_mask'],digit_completion_mask
            else:
                return per_token_logps,mini_batch_encoded_inputs['completion_mask']
        else:
            return per_token_logps


    def get_in_context_done(self,task:str,first_image:List[Image.Image],n_pre_image:List[Image.Image],now_image:List[Image.Image],ref_image_list:List[Image.Image],ref_num=9,rich=False):
        """
        In context 的方式获取done
        提供一段参考ref_image_list
        first_image是你的轨迹的第一张图,这里为list是可以batch推理
        n_pre_image是上一个时刻
        now_image是当前时刻
        """
        if ref_image_list is not None:
            ref_images=[ref_image_list[0]]
            delta=(len(ref_image_list)-1)/(ref_num-1)
            for i in range(1,ref_num):
                ref_images.append(ref_image_list[int(i*delta)])
        else:
            ref_num=0
        one_prompt=self.get_score_prompt(task=task,trajectory_len=ref_num,think=True)
        batch_prompt=[]
        batch_image=[]
        for i in range(len(now_image)):
            batch_prompt.append(one_prompt)
            batch_image.append(ref_images+[first_image[i],n_pre_image[i],now_image[i]])
        infer_requests=self.get_infer_requests(prompt=batch_prompt,images=batch_image)
        response_list,infer_time=self.chat(infer_requests)
        answers_list,complete_requests_list=self.results_format(response_list,infer_requests,rich=rich)
        think_pre_value_list=[]
        think_post_value_list=[]
        think_critic_list=[]
        for one in answers_list:
            one_critic=one.split('</think>')[1]
            one_pre_value=one.split('first image progressing: ')[1].split('%')[0]
            one_post_value=one.split('second image progressing: ')[1].split('%')[0]
            think_critic_list.append("{:.1f}".format((100-float(one_pre_value))*float(one_critic)/100.0))
            think_pre_value_list.append(one_pre_value)
            think_post_value_list.append("{:.3f}".format(float(one_post_value)/100.0))
        batch_done=think_post_value_list
        batch_related_critic=think_critic_list
        return batch_done,batch_related_critic

    def eval_trajectory(self, task,
                            batch_num=20, ref_num=9, skip=1, frame_skip=False, done_threshold=False, 
                            video_output=False, output_path=""):
        list_video = task.frames
        ref_list = task.ref_frames
        done_ref = ref_list
        critic_list, think_value_list=self.get_trajectory_critic(
            task=task.description,
            image_list=list_video,
            ref_image_list=ref_list,
            batch_num=batch_num,
            ref_num=ref_num,
            think=False,
            skip=skip,
            rich=False,
            reverse_eval=False,
            frame_skip=frame_skip,
            addition_scale=1,
            bias=0,
            positive_clip=0,
            negative_clip=0,
            related_critic=False,
        )
        value_list = think_value_list
        # if frame_skip:
        done_list = value_list
        if video_output:
            if ref_num>0:
                ref_image_paths=ref_list
            else:
                ref_image_paths=None
            fps = 1
            video_path = video_trajectory(
                traj_id=task.video_name[:-4],
                view='0',
                image_paths=[],
                image_objects=list_video,
                done_list=done_list,
                critic_list=critic_list,
                value_list=value_list,
                task=task.description,
                output_path=output_path,
                fps=fps,
                ref_image_paths=ref_image_paths,
                n_num=ref_num
            )
        
        return done_list
            
    def web_trajectory_critic(self, task_description, main_video_path, reference_video_path=None, 
                            batch_num=20, ref_num=9, think=False, skip=1, rich=False, reverse_eval=False,output_path=None,fps=None,frame_skip=False,addition_scale=1,bias=0,positive_clip=0,negative_clip=0,related_critic=False,done_flag=False,in_context_done=False,done_threshold=False,video_output=True):
        list_video=images_get_from_video(main_video_path)
        ref_list=None
        if reference_video_path is None:
            ref_num=0
        else:
            ref_list=images_get_from_video(reference_video_path)
        if done_flag:
            if in_context_done:
                done_ref=ref_list
            else:
                done_ref=None
            done_list=self.get_trajectory_done(task=task_description,image_list=list_video,ref_image_list=done_ref,batch_num=batch_num,rich=True,ref_num=ref_num,threshold=done_threshold)
            if frame_skip:
                done_list=[done_list[i] for i in range(0,len(done_list),skip)]
        else:
            done_list=None
        critic_list, think_value_list=self.get_trajectory_critic(task=task_description,image_list=list_video,ref_image_list=ref_list,batch_num=batch_num,ref_num=ref_num,think=think,skip=skip,rich=rich,reverse_eval=reverse_eval,frame_skip=frame_skip,addition_scale=addition_scale,bias=bias,positive_clip=positive_clip,negative_clip=negative_clip,related_critic=related_critic)
        value_list=think_value_list
        if video_output:
            if ref_num>0:
                ref_image_paths=ref_list
            else:
                ref_image_paths=None
            if fps:
                pass
            else:
                fps=5.0/skip
            video_path = video_trajectory(
                traj_id='temp',
                view='0',
                image_paths=[],
                image_objects=list_video,
                done_list=done_list,
                critic_list=critic_list,
                value_list=value_list,
                task=task_description,
                output_path=output_path,
                fps=fps,
                ref_image_paths=ref_image_paths,
                n_num=ref_num
            )
        else:
            video_path=None
        return video_path,value_list,critic_list,done_list
    
    def web_trajectory_done(self, task_description, main_image_path, reference_image_path=None, rich=False):
        done_list=self.get_trajectory_done(task=task_description,image_list=[main_image_path],batch_num=1,rich=rich)
        return f"Task: {task_description}\nCritic Score: {done_list[0]}"
    

    def get_trajectory_task(self, image_list):
        one_prompt=self.get_task_prompt()
        infer_requests=self.get_infer_requests(prompt=one_prompt,images=[image_list[0],image_list[-1]])
        response_list,infer_time=self.chat(infer_requests)
        answers_list,complete_requests_list=self.results_format(response_list,infer_requests)
        return answers_list

    
    def get_trajectory_done(self,task:str,image_list:List[Image.Image],ref_image_list:List[Image.Image]=None,batch_num:int=20,rich=False,ref_num=9,threshold=0,skip=1,goal_image:Image.Image=None):
        """
        输入一条trajectory的所有图片,输出每张图片的done,0~1的突变值
        当有ref_image_list时,进行过程判断,输出0~1的渐变值
        当有gaol_image时,进行最终状态判断,输出0~1的突变值(与当有ref_image_list冲突),task=None时只依靠图片
        """
        batch_prompt=[]
        batch_image=[]
        done_list=[]
        if skip>1:
            image_list=[image_list[i] for i in range(0,len(image_list),skip)]
        if ref_image_list is None and goal_image is None:
            for i in tqdm.tqdm(range(len(image_list)),desc='done processing'):
                one_prompt=self.get_done_prompt(task=task)
                batch_prompt.append(one_prompt)
                batch_image.append([image_list[i]])
                if (i+1) % batch_num==0 or i==len(image_list)-1:
                    infer_requests=self.get_infer_requests(prompt=batch_prompt,images=batch_image)
                    response_list,infer_time=self.chat(infer_requests)
                    answers_list,complete_requests_list=self.results_format(response_list,infer_requests,rich=rich)
                    print(f'infer_time:{infer_time}s')
                    print(f'answers_list:{answers_list}')
                    done_list.extend(answers_list)
                    batch_prompt=[]
                    batch_image=[]
        elif ref_image_list:
            first_image=[]
            n_pre_image=[]
            now_image=[]
            for i in tqdm.tqdm(range(1,len(image_list)),desc='done processing'):
                first_image.append(image_list[0])
                n_pre_image.append(image_list[i-1])
                now_image.append(image_list[i])
                if (i+1) % batch_num==0 or i==len(image_list)-1:
                    batch_done,batch_related_critic=self.get_in_context_done(task=task,first_image=first_image,n_pre_image=n_pre_image,now_image=now_image,ref_image_list=ref_image_list,ref_num=ref_num,rich=rich)
                    print(f'answers_list:{batch_done}')
                    done_list.extend(batch_done)
                    first_image=[]
                    n_pre_image=[]
                    now_image=[]
            done_list=[0]+done_list
        else:
            for i in tqdm.tqdm(range(len(image_list)),desc='done processing'):
                one_prompt=self.get_in_context_done_prompt(task=task)
                batch_prompt.append(one_prompt)
                batch_image.append([goal_image,image_list[i]])
                if (i+1) % batch_num==0 or i==len(image_list)-1:
                    infer_requests=self.get_infer_requests(prompt=batch_prompt,images=batch_image)
                    response_list,infer_time=self.chat(infer_requests)
                    answers_list,complete_requests_list=self.results_format(response_list,infer_requests,rich=rich)
                    print(f'infer_time:{infer_time}s')
                    print(f'answers_list:{answers_list}')
                    done_list.extend(answers_list)
                    batch_prompt=[]
                    batch_image=[]
        done_list = [float(one) if float(one) > threshold else 0.0 for one in done_list]
        return done_list

    def get_trajectory_critic(self,task:str,image_list:List[Image.Image],ref_image_list:List[Image.Image]=None,batch_num:int=20,ref_num=9,think=False,skip=1,rich=False,reverse_eval=False,frame_skip=True,addition_scale=1,bias=0,related_critic=False,positive_clip=0,negative_clip=0,value_simple=True):
        """
        输入一条trajectory的所有图片,输出每张图片的critic和processing value
        可以给一条参考轨迹
        """
        batch_prompt=[]
        batch_image=[]
        critic_list=[]
        value_list=[]
        if ref_image_list is not None:
            ref_images=[ref_image_list[0]]
            delta=(len(ref_image_list)-1)/(ref_num-1)
            for i in range(1,ref_num):
                ref_images.append(ref_image_list[int(i*delta)])
        else:
            ref_num=0
        if frame_skip:
            select_idx=range(skip,len(image_list),skip)
        else:
            select_idx=range(skip,len(image_list))
        for i in tqdm.tqdm(select_idx,desc='critic processing'):
            one_prompt=self.get_score_prompt(task=task,trajectory_len=ref_num,think=think)
            batch_prompt.append(one_prompt)
            if ref_image_list is not None:
                if reverse_eval:
                    batch_image.append(ref_images+[image_list[0],image_list[i],image_list[i-skip]])
                else:
                    batch_image.append(ref_images+[image_list[0],image_list[i-skip],image_list[i]])
            else:
                if reverse_eval:
                    batch_image.append([image_list[i],image_list[i-skip]])
                else:
                    batch_image.append([image_list[i-skip],image_list[i]])
            if (len(batch_prompt)) % batch_num==0 or len(critic_list)+len(batch_prompt)==len(select_idx):
                infer_requests=self.get_infer_requests(prompt=batch_prompt,images=batch_image)
                response_list,infer_time=self.chat(infer_requests)
                answers_list,complete_requests_list=self.results_format(response_list,infer_requests,rich=rich)
                print(f'infer_time:{infer_time}s')
                print(f'answers_list:{answers_list}')
                critic_list.extend(answers_list)
                batch_prompt=[]
                batch_image=[]
        if think:
            think_pre_value_list=[]
            think_post_value_list=[]
            think_critic_list=[]
            for one in critic_list:
                one_critic=one.split('</think>')[1]
                one_pre_value=one.split('first image progressing: ')[1].split('%')[0]
                one_post_value=one.split('second image progressing: ')[1].split('%')[0]
                think_critic_list.append(one_critic)
                think_pre_value_list.append(one_pre_value)
                think_post_value_list.append(one_post_value)
            if reverse_eval:
                temp=think_pre_value_list
                think_pre_value_list=think_post_value_list
                think_post_value_list=temp
                think_critic_list=[0-float(one) for one in think_critic_list]
            think_pre_value_list=think_pre_value_list+think_post_value_list[-skip:]
            think_post_value_list=think_pre_value_list[:skip]+think_post_value_list
            critic_list=think_critic_list
            if frame_skip:
                pass
            else:
                critic_list=[float(one)/skip for one in critic_list]
            critic_list=[float(one)/addition_scale for one in critic_list]
            value_list=self.critic_to_value_simple(critic_list,simple=value_simple)
        else:
            if reverse_eval:
                critic_list=[0-float(one) for one in critic_list]
            if frame_skip:
                pass
            else:
                critic_list=[float(one)/skip for one in critic_list]
            critic_list=[float(one)/addition_scale for one in critic_list]
            value_list=self.critic_to_value_simple(critic_list,simple=value_simple)
        if related_critic:
            critic_list=[value_list[i]-value_list[i-1] for i in range(1,len(value_list))]
        if bias!=0:
            critic_list=[one+bias for one in critic_list]
        if positive_clip!=0:
            critic_list=[one if (one<0 or one>positive_clip) else 0 for one in critic_list]
        if negative_clip!=0:
            critic_list=[one if (one>0 or one<-negative_clip) else 0 for one in critic_list]
        if related_critic:
            value_list=[0]
            for one in critic_list:
                value_list.append(value_list[-1]+one)
        else:
            value_list=self.critic_to_value_simple(critic_list,simple=value_simple)
        #这里的value也是done，越接近100完成度越高
        return critic_list,value_list

    def magic_smooth(self,task,file_path,hz=10,value_simple='mix_f',ref_image_list=None,ref_num=9,think=False,max_skip=3):
        import os
        import pickle
        import gzip
        import subprocess
        from video_tool import read_data_and_create_video,images_get_from_video
        if type(ref_image_list)  == str:
            if ref_image_list.endswith(".pkl.gz"):
                ref_image_list = read_data_and_create_video(ref_image_list)
            else:
                ref_image_list=images_get_from_video(ref_image_list)
        image_list=read_data_and_create_video(file_path, camera_name=None, create_video=False, output_path=None, fps=hz, start_frame=0, end_frame=None)
        skip=int(hz/2)
        critic_list,value_list=self.get_trajectory_critic(task=task,image_list=image_list,ref_image_list=ref_image_list,batch_num=10,ref_num=ref_num,think=think,skip=skip,rich=True,reverse_eval=False,frame_skip=True,addition_scale=1,bias=0,related_critic=False,positive_clip=0,negative_clip=0,value_simple=value_simple)

        skip_idx_range=[]
        skip_idxs=[]
        i=0
        while i <len(critic_list):
            if float(critic_list[i])<=0:
                skip_window=2
                while i+skip_window<len(critic_list) and skip_window<=max_skip:
                    two_image=[image_list[0],image_list[i*skip],image_list[(i+skip_window)*skip]]
                    one_critic_list,_=self.get_trajectory_critic(task=task,image_list=two_image,ref_image_list=ref_image_list,batch_num=10,ref_num=ref_num,think=think,skip=1,rich=True,reverse_eval=False,frame_skip=True,addition_scale=1,bias=0,related_critic=False,positive_clip=0,negative_clip=0,value_simple=value_simple)
                    if float(one_critic_list[-1])<0:
                        skip_window+=1
                    else:
                        break
                skip_idx_range.append([i,i+skip_window])
                skip_idxs.extend(range(i*skip+1,(i+skip_window)*skip))
                i+=skip_window
                print("skip_window:",skip_window)
            else:
                i+=1
        with gzip.open(file_path, 'rb') as f:
            original_data = pickle.load(f)
        
        base_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0]  # 去掉.pkl.gz
        output_dir = os.path.join(base_dir, f"{file_name}_smoothed")
        os.makedirs(output_dir, exist_ok=True)
        new_data=[one for i,one in enumerate(original_data) if i not in skip_idxs]
        new_images=[one for i,one in enumerate(image_list) if i not in skip_idxs]

        data_file = os.path.join(output_dir, f"{file_name}_smoothed.pkl.gz")
            
        with gzip.open(data_file, 'wb') as f:
            pickle.dump(new_data, f)
        
        video_file = os.path.join(output_dir, f"{file_name}_smoothed.mp4")
        
        temp_dir = os.path.join(output_dir, f"temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            for j, img in enumerate(new_images):
                frame_file = os.path.join(temp_dir, f"frame_{j:06d}.png")
                if hasattr(img, 'save'):
                    img.save(frame_file)
                else:
                    from PIL import Image
                    if isinstance(img, np.ndarray):
                        Image.fromarray(img).save(frame_file)
                    else:
                        Image.fromarray(np.array(img)).save(frame_file)
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(hz),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                video_file
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"Created video: {video_file}")
            
        except Exception as e:
            print(f"Error creating video for segment {i}: {e}")
            
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        return new_data
        



    def magic_split(self,task,file_path,hz=10,value_simple='mix_f',done_assist=False,ref_image_list=None):
        import os
        import pickle
        import gzip
        import subprocess
        from video_tool import read_data_and_create_video,images_get_from_video
        from magic_detect import adaptive_peak_valley_detection,plot_results
        if type(ref_image_list)  == str:
            if ref_image_list.endswith(".pkl.gz"):
                ref_image_list = read_data_and_create_video(ref_image_list)
            else:
                ref_image_list=images_get_from_video(ref_image_list)
        
        skip=int(hz/2)
        done_len=4
        gap_merge_len=4#一般看数据的平均记录周期
        image_list=read_data_and_create_video(file_path, camera_name=None, create_video=False, output_path=None, fps=hz, start_frame=0, end_frame=None)
        critic_list,value_list=self.get_trajectory_critic(task=task,image_list=image_list,ref_image_list=ref_image_list,batch_num=10,ref_num=9,think=False,skip=skip,rich=True,reverse_eval=False,frame_skip=True,addition_scale=1,bias=0,related_critic=False,positive_clip=0,negative_clip=0,value_simple=value_simple)
        
        if done_assist:
            done_list=self.get_trajectory_done(task=task,image_list=image_list,ref_image_list=None,batch_num=10,ref_num=2,rich=True,threshold=0.9,skip=skip)
            assist_peaks={}
            peak_window=[]
            done_list=done_list+[0]
            for k in range(len(done_list)):
                if done_list[k]>=0.95:
                    peak_window.append(k)
                else:
                    if len(peak_window)>=done_len:
                        assist_peaks[peak_window[done_len-1]]=peak_window
                    peak_window=[]
            peak_list=list(assist_peaks.keys())
            index=0
            while index<len(peak_list)-1:
                if assist_peaks[peak_list[index+1]][0]-assist_peaks[peak_list[index]][-1]<=gap_merge_len:
                    assist_peaks[peak_list[index]].extend(assist_peaks[peak_list[index+1]])
                    del assist_peaks[peak_list[index+1]]
                    del peak_list[index+1]
                else:
                    index+=1
            result = adaptive_peak_valley_detection(
                        value_list, 
                        assist_peaks=peak_list,
                        window_size=None,  # Adaptive window size
                        verbose=True,
                        min_distance=10,
                        prominence_threshold=0.1,
                        outlier_sensitivity=3
            )
        else:
            result = adaptive_peak_valley_detection(
                        value_list, 
                        window_size=None,  # Adaptive window size
                        verbose=True,
                        min_distance=10,
                        prominence_threshold=0.01,
                        outlier_sensitivity=1.5
            )
        

        with gzip.open(file_path, 'rb') as f:
            original_data = pickle.load(f)
        
        base_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0]  # 去掉.pkl.gz
        output_dir = os.path.join(base_dir, f"{file_name}_segments")
        os.makedirs(output_dir, exist_ok=True)
        plot_results(value_list, result,save_path=os.path.join(output_dir, "peak_valley.png"))
        peaks = [one*skip for one in result['peaks']]
        valleys = [one*skip for one in result['valleys']]
        
        ascending_dir = os.path.join(output_dir, "ascending")
        descending_dir = os.path.join(output_dir, "descending")
        os.makedirs(ascending_dir, exist_ok=True)
        os.makedirs(descending_dir, exist_ok=True)
        
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        all_points = []
        for p in peaks:
            all_points.append((p, 'peak'))
        for v in valleys:
            all_points.append((v, 'valley'))
        all_points.sort(key=lambda x: x[0])
        
        segments = []
        
        for i, (point_idx, point_type) in enumerate(all_points):
            if i == 0:
                prev_type='valley'
                start_idx = 0
            else:
                start_idx=all_points[i-1][0]
                prev_type = all_points[i-1][1]

            end_idx = point_idx
            if prev_type == 'valley' and point_type == 'peak':
                segment_type = 'ascending'
            elif prev_type == 'peak' and point_type == 'valley':
                segment_type = 'descending'
            else:
                continue
                
            data_segment = original_data[start_idx:end_idx]
            image_segment = image_list[start_idx:end_idx]
            
            segments.append({
                'type': segment_type,
                'start': start_idx,
                'end': end_idx,
                'data': data_segment,
                'images': image_segment
            })
            
        
        if segments and segments[-1]['type'] == 'descending':
            segments.pop()
        
        for i, segment in enumerate(segments):
            segment_type = segment['type']
            
            task_name = self.get_trajectory_task(segment['images'])[0].split('</task>')[0].strip().replace(' ', '_').replace('.', '_').replace(',', '_')
            
            if segment_type == 'ascending':
                data_file = os.path.join(ascending_dir, f"{task_name}_{i:03d}.pkl.gz")
            else:
                data_file = os.path.join(descending_dir, f"{task_name}_{i:03d}.pkl.gz")
                
            with gzip.open(data_file, 'wb') as f:
                pickle.dump(segment['data'], f)
            
            video_file = os.path.join(video_dir, f"{segment_type}_{task_name}_{i:03d}.mp4")
            
            temp_dir = os.path.join(output_dir, f"temp_{i}")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                for j, img in enumerate(segment['images']):
                    frame_file = os.path.join(temp_dir, f"frame_{j:06d}.png")
                    if hasattr(img, 'save'):
                        img.save(frame_file)
                    else:
                        from PIL import Image
                        if isinstance(img, np.ndarray):
                            Image.fromarray(img).save(frame_file)
                        else:
                            Image.fromarray(np.array(img)).save(frame_file)
                
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(hz),
                    '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    video_file
                ]
                
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Created video: {video_file}")
                
            except Exception as e:
                print(f"Error creating video for segment {i}: {e}")
                
            finally:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        print(f"Processing complete. Segments saved to: {output_dir}")
        print(f"Total segments: {len(segments)}")
        print(f"Ascending segments: {sum(1 for s in segments if s['type'] == 'ascending')}")
        print(f"Descending segments: {sum(1 for s in segments if s['type'] == 'descending')}")
        
        return segments
    
    def critic_to_value_simple(self,critic_list,simple=True):
        """
        将critic计算为0-100的value,输入需是一条轨迹按顺序的critic
        """
        value_list=[0]
        for i in range(len(critic_list)):
            if float(critic_list[i])>0 or simple is True:
                value_list.append(value_list[-1]+(100-value_list[-1])*float(critic_list[i])/100.0)
            else:
                if simple=='mix_f':
                    if value_list[-1]>50:
                        value_list.append(max(10,100-max((100-value_list[-1]),1.0)/(100+float(critic_list[i]))*100.0))
                    else:
                        value_list.append(value_list[-1]+value_list[-1]*float(critic_list[i])/100.0)
                else:
                    value_list.append(100-max((100-value_list[-1]),1.0)/(100+float(critic_list[i]))*100.0)
        return value_list
    
    def compute_voc(self,values_list):
        """
        计算Value-Order Correlation (VOC)
        
        参数:
            predicted_values: 模型预测的值序列，形状为(T,)，其中T是时间步数
            
        返回:
            voc: Value-Order Correlation值，范围从-1到1
        """
        T = len(values_list)
        time_order = np.arange(T)
        
        correlation, _ = spearmanr(values_list, time_order)
        return correlation
    
    def compute_negative_rate(self,critic_list):
        """
        计算逆序动作的比例
        """
        negative_critic=[one for one in critic_list if one<0]

        return float(len(negative_critic))/len(critic_list)
  
from swift.llm import InferRequest, InferClient, RequestConfig
from typing import List, Union, Sequence, Optional
from io import BytesIO
import base64

def _pil_to_dataurl(pil: Image.Image, fmt="JPEG", quality=90) -> str:
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = BytesIO()
    pil.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"

def _bytes_to_video_dataurl(data: bytes, mime: str = "video/mp4") -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

  
class GAC_model_client(nn.Module):
    def __init__(self,tag='critic'):
        self.tag=tag
        self.temperature=1
        self.max_tokens=10240
        self.do_sample=True
        self.top_logprobs=10
        self.logprobs=True
        self.top_k=None
        self.system_prompt=None
        self.action_data={}
        self.critic_data={}
        self.sft_dataset=None
        self.dataclient=DataProcessor()
        self.dataclient.prompt_templete['v3']="Image-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image. + score means the second image is closer, - score means the first image is closer\nResponse the relative progressing of target task follow <score>. The target task is: <task> {} </task> <score>"
        self.dataclient.prompt_templete['v3_think']="0% <image>\nThis image is the trajectory beginning of the following two images\nImage-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image. + score means the second image is closer, - score means the first image is closer\nResponse the relative progressing of target task follow <score>. The target task is: <task> {} </task> <score>"
        

    def _songling_process(self,action):
        action=copy.deepcopy(action)
        for i in range(len(action)):
            if i in [0, 1, 2]:
                action[i]=action[i]//1000
            elif i in [3,4,5]:
                action[i]=action[i]//1000
            elif i ==6:
                pass
            else:
                pass
        return action

    def format_state(self,state,gripper_format=False):
        state=self._songling_process(state)
        if gripper_format:
            state_open=round((state[6]//10000)/7.0,1)
            state[6]=state_open
        else:
            state[6]=state[6]//1000
        return state

    def get_score_prompt(self,task,trajectory_len=0,think=False):
        "two or len+3 image"
        if trajectory_len>0:
            trajectory_prompt=trojectory_example_prompt(list(range(trajectory_len)),task=task)
            full_prompt=trajectory_prompt+self.dataclient.prompt_templete['v3_think'].format(task)
        else:
            full_prompt=self.dataclient.prompt_templete['v3'].format(task)
        if think:
            full_prompt+=self.dataclient.prompt_templete['think']
        return full_prompt
    
    def get_done_prompt(self,task):
        "one image"
        return self.dataclient.prompt_templete["task_done"].format(task)
    
    def get_in_context_done_prompt(self,task=None):
        "two image"
        if task:
            return self.dataclient.prompt_templete["context_task_done"].format(task,task)
        else:
            return self.dataclient.prompt_templete["image_done"]
    
    def get_task_prompt(self):
        "two image"
        return self.dataclient.prompt_templete["task_vqa"]
    
    def get_action_inverse_prompt(self):
        "two image"
        return self.dataclient.prompt_templete["action_inverse"]
    
    def get_action_score_prompt(self,task,state,action):
        "one image"
        format_fuction=self.dataclient.action_format['songling']
        action_score_prompt=self.dataclient.prompt_templete['task_action_score'].format(format_fuction(state),format_fuction(action),task)
        return action_score_prompt

    def get_action_prompt(self,task,view_num=1,position_output=False,simple=False,state='type',output_num=1,think=False):
        if simple:
            format_fuction=self.dataclient.action_format_simple['songling']
            action_key='task_action_simple'
            if position_output:
                action_key='task_position_simple'
        else:
            format_fuction=self.dataclient.action_format['songling']
            action_key='task_action'
            if position_output:
                action_key='task_position'
        "view_num(max 3) image"
        image_prompt=self.dataclient.image_prompt_templete[view_num]
        full_prompt=image_prompt+self.dataclient.prompt_templete[action_key].format(format_fuction(state,state=position_output),task)
        if output_num>1:
            full_prompt+=f'*{output_num}'
        if think:
            full_prompt+=self.dataclient.prompt_templete['think']
        return full_prompt
    
    def get_fast_action_prompt(self,task,view_num=1,position_output=False,simple=False,state='type',output_num=1,think=False):
        full_prompt = self.get_action_prompt(task,view_num,position_output,simple,state,output_num,think)
        full_prompt+='<chunks>'
        return full_prompt
    
    def set_config(self):
        self.request_config = RequestConfig(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )

    def _get_internvl2_per_token_logps(self, model, inputs):
        from trl.trainer.utils import selective_log_softmax
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        inputs = {
            k: v
            for k, v in inputs.items() if k not in
            ['logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps']
        }
        _, inputs = self.template.pre_forward_hook(self.model, None, inputs)
        logits = model(**inputs).logits
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        logits = logits / self.temperature
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)
     
    def init_model(
        self,
        model_path: str | None,
        model_type: str = "internvl2",
        device_map: str | None = None,
        torch_dtype=torch.bfloat16,
        adapter: str | None = None,
        use_server: bool = True,           # 关键：server 模式不加载本地大模型
    ):
        """
        use_server=True 时：
          - 不加载本地权重，不构建 PtEngine
          - 只拿到模板（如需 tokenizer，仅加载 tokenizer）
        use_server=False 时：
          - 本地推理：加载模型并构建 PtEngine
        """
        from swift.llm.model.register import get_model_tokenizer
        from swift.llm.template import get_template
        from swift.plugin import InferStats

        if use_server:
            # 只需要 template；尽量避免加载大模型
            # 很多模板并不需要 tokenizer；如果需要，可以只加载 tokenizer：
            tokenizer = None
            try:
                _, tokenizer = get_model_tokenizer(
                    model_id_or_path=model_path,
                    model_type=model_type,
                    torch_dtype=torch_dtype,
                    device_map=None,           # 禁止把模型绑到本地 GPU
                    attn_impl='flash_attn',
                    load_model=False,          # 如果 swift 版本支持
                )
            except Exception:
                tokenizer = None  # 模板能否与 None 搭配取决于 swift 版本

            self.template = get_template(model_type, tokenizer)
            self.infer_stats = InferStats()
            self.engine = None  # 服务器推理，无本地引擎

        else:
            # 本地 PtEngine 模式（可选）
            self.model, tokenizer = get_model_tokenizer(
                model_id_or_path=model_path,
                model_type=model_type,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_impl='flash_attn',
            )
            from swift.llm import PtEngine
            self.template = get_template(model_type, tokenizer)
            self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)
            self.infer_stats = InferStats()

    def bind_gateway(self, rm_gateway):
        """绑定共享的 Ray RMGateway 句柄"""
        self.rm_gateway = rm_gateway
   
    def chat(self,infer_requests):
        start_t=time.time()
        response_list = self.engine.infer(
        infer_requests, request_config=self.request_config, metrics=[self.infer_stats])
        end_t=time.time()
        infer_time=end_t-start_t
        return response_list,infer_time
    
    def results_format(self,response_list,infer_requests,rich=False):
        infer_requests=copy.deepcopy(infer_requests)
        answers=[]
        for i in range(len(response_list)):
            if rich:
                rich_answer=''
                for one in response_list[i].choices[0].logprobs['content'][:-1]:
                    if one['token'].isdigit():
                        temp_num=0
                        temp_weight=0
                        top_prob=math.e**one['top_logprobs'][0]['logprob']
                        for one_tops in one['top_logprobs']:
                            top_num=one_tops['token']
                            if top_num.isdigit():
                                prob=math.e**one_tops['logprob']
                                if prob>top_prob*0.1:
                                    temp_num+=float(top_num)*prob
                                    temp_weight+=prob
                        rich_answer+="{:.1f}".format(temp_num/temp_weight)
                    else:
                        rich_answer+=one['token']
                answers.append(rich_answer)
            else:
                answers.append(response_list[i].choices[0].message.content)
            infer_requests[i].messages.append({'role': 'assistant', 'content': response_list[i].choices[0].message.content})
        return answers,infer_requests
    
    def fast_results_format(self,response_list,infer_requests,tokenizer,time_horizon=10,action_dim=7):
        infer_requests=copy.deepcopy(infer_requests)
        answers=[]
        for i in range(len(response_list)):
            temp=[]
            text=[]
            for one in response_list[i].choices[0].logprobs['content'][:-1]:
                ids=tokenizer.encode(one['token'],add_special_tokens=False)
                if 92537-ids[0]<=2048:
                    temp.extend(ids)
            temp=[[92537-one for one in temp]]
            fast_chunk=tokenizer.fasttokenizer.decode(temp,time_horizon=time_horizon,action_dim=action_dim)
            infer_requests[i].messages.append({'role': 'assistant', 'content': response_list[i].choices[0].message.content})
        return fast_chunk,infer_requests
    
    def denormalize_fastchunk(self,fast_chunk,params):
        return denormalize_with_params(fast_chunk,params)

    def set_system_prompt(self,system_prompt=None):
        if system_prompt is not None:
            self.system_prompt=system_prompt
        elif system_prompt=='default':
            self.system_prompt=None
        else:
            self.system_prompt='You are a visual-language assistant designed to interpret spatial and task-related information from images and text. Provide precise, context-aware responses and actionable guidance to assist in achieving task objectives.'
    
    def _process_image_to_pil(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        将各种格式的图像输入转换为448x448的PIL.Image
        
        Args:
            image_input: 图像URL、文件路径、base64编码字符串或PIL.Image对象
        
        Returns:
            PIL.Image: 调整为448x448尺寸的PIL图像
        """
        pil_image = None
        
        if isinstance(image_input, Image.Image):
            pil_image = image_input
        elif isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input)
                pil_image = Image.open(BytesIO(response.content))
            elif image_input.startswith('data:image'):
                header, encoded = image_input.split(',', 1)
                image_data = base64.b64decode(encoded)
                pil_image = Image.open(BytesIO(image_data))
            elif len(image_input) > 100 and not '/' in image_input and not '\\' in image_input:
                try:
                    image_data = base64.b64decode(image_input)
                    pil_image = Image.open(BytesIO(image_data))
                except:
                    pil_image = Image.open(image_input)
            else:
                pil_image = Image.open(image_input)
                
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            if arr.dtype != np.uint8:
                # 支持 [0,1] 或 [0,255] 浮点
                arr = np.asarray(arr)
                if arr.dtype.kind in ('f', 'c'):  # float/complex
                    a = np.clip(arr.real, 0, 1) if arr.max() <= 1.0 else np.clip(arr.real, 0, 255) / 255.0
                    arr = (a * 255.0).round().astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)

            # 形状处理: HxW、HxWxC 或 CxHxW
            if arr.ndim == 2:
                # 灰度
                pil_image = Image.fromarray(arr, mode='L').convert('RGB')
            elif arr.ndim == 3:
                if arr.shape[0] in (1, 3, 4) and arr.shape[0] < 8:
                    # CxHxW -> HxWxC
                    arr = np.transpose(arr, (1, 2, 0))
                # 现在应是 HxWxC
                c = arr.shape[2]
                if c == 1:
                    pil_image = Image.fromarray(arr.squeeze(-1), mode='L').convert('RGB')
                elif c == 3:
                    # 默认按 RGB 解释（如果来源是 OpenCV 的 BGR，你可在上游转换或在此处反转通道）
                    pil_image = Image.fromarray(arr, mode='RGB')
                elif c == 4:
                    pil_image = Image.fromarray(arr, mode='RGBA').convert('RGB')
                else:
                    raise ValueError(f"不支持的通道数: {c}, 期望 1/3/4")
            else:
                raise ValueError(f"不支持的 ndarray 维度: {arr.ndim}, 期望 2 或 3")
        else:
            raise ValueError(f"不支持的图像输入类型: {type(image_input)}")
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        pil_image = pil_image.resize((448, 448), Image.Resampling.LANCZOS)
        
        return pil_image
    

    def get_infer_requests_client(
        self,
        prompts: Union[str, Sequence[str]],
        images: Optional[
            Union[
                Image.Image, str, bytes, "np.ndarray", "torch.Tensor",
                Sequence[Union[Image.Image, str, bytes, "np.ndarray", "torch.Tensor"]],
                Sequence[Sequence[Union[Image.Image, str, bytes, "np.ndarray", "torch.Tensor"]]]
            ]
        ] = None,
        videos: Optional[Union[bytes, str, Sequence[Union[bytes, str]], Sequence[Sequence[Union[bytes, str]]]]] = None,
    ) -> List[InferRequest]:
        """
        返回: List[InferRequest]，每条请求 messages 里包含:
        - system（可选）
        - user: content=[{"type":"text","text":...}, {"type":"image_url","image_url":...}*, {"type":"video","video":...}*]
        images: 支持 PIL/路径或URL/base64/bytes/np.ndarray/torch.Tensor；内部统一 resize 到 448 并转 data URL
        videos: 支持 bytes 或本地路径（自动读 bytes），转 data URL（video/mp4）
        """
        # 归一化 batch 维度
        if isinstance(prompts, str):
            prompts = [prompts]
            images_list = [images]
            videos_list = [videos]
        else:
            prompts = list(prompts)
            # 对齐 images
            # if images is None:
            # elif isinstance(images, (list, tuple)) and len(images) == len(prompts) and not (
            # ):
            images_list = images

            # 对齐 videos
            if videos is None:
                videos_list = [None] * len(prompts)
            elif isinstance(videos, (list, tuple)) and len(videos) == len(prompts) and not (
                len(videos) > 0 and isinstance(videos[0], (list, tuple))
            ):
                videos_list = list(videos)
            else:
                videos_list = [videos] * len(prompts)

        reqs: List[InferRequest] = []
        for i, txt in enumerate(prompts):
            # system
            messages = []
            if getattr(self, "system_prompt", None):
                messages.append({"role": "system", "content": self.system_prompt})

            # user content
            content = []
            if txt is not None and txt != "":
                content.append({"type": "text", "text": str(txt)})

            # images -> data URL
            imgs = images_list[i]
            if imgs is not None:
                if not isinstance(imgs, (list, tuple)):
                    imgs = [imgs]
                for img in imgs:
                    if img is None:
                        continue
                    pil = self._process_image_to_pil(img)  # 你已实现：兼容 np/torch/bytes/str，统一到 RGB 448
                    content.append({"type": "image_url", "image_url": _pil_to_dataurl(pil)})

            # videos -> data URL
            vs = videos_list[i]
            if vs is not None:
                if not isinstance(vs, (list, tuple)):
                    vs = [vs]
                for v in vs:
                    if v is None:
                        continue
                    if isinstance(v, bytes):
                        v_bytes = v
                    else:
                        # 当作本地路径读；（如果是 http(s) URL，你可以先用 requests.get 拿 bytes）
                        with open(os.fspath(v), "rb") as f:
                            v_bytes = f.read()
                    content.append({"type": "video", "video": _bytes_to_video_dataurl(v_bytes, "video/mp4")})

            messages.append({"role": "user", "content": content})
            reqs.append(InferRequest(messages=messages))

        return reqs
    
    
    def get_logprobs(self,infer_requests,return_mask=False,digit=False):
        old_mode=self.template.mode
        self.template.set_mode('train')
        mini_batch_encoded_inputs = [self.template.encode(infer_request) for infer_request in infer_requests]
        mini_batch_encoded_inputs = to_device(
            self.template.data_collator(mini_batch_encoded_inputs), self.model.device)
        labels = mini_batch_encoded_inputs.pop('labels')
        logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
        mini_batch_encoded_inputs['logits_to_keep'] = logits_to_keep
        mini_batch_encoded_inputs['completion_mask'] = labels[:, -logits_to_keep:] != -100
        per_token_logps = self._get_internvl2_per_token_logps(self.model, mini_batch_encoded_inputs)
        self.template.set_mode(old_mode)
        if return_mask:
            if digit:
                tokenizer = self.template.tokenizer
                
                digit_completion_mask = torch.zeros_like(labels[:, -logits_to_keep:], dtype=torch.bool)
                for i in range(labels.shape[0]):
                    for j in range(labels.shape[1] - logits_to_keep, labels.shape[1]):
                        if labels[i, j] != -100:
                            token_id = labels[i, j].item()
                            token_str = tokenizer.decode([token_id]).strip()
                            if re.search(r'[0-9+\-]', token_str):
                                digit_completion_mask[i, j - (labels.shape[1] - logits_to_keep)] = True
                return per_token_logps, mini_batch_encoded_inputs['completion_mask'],digit_completion_mask
            else:
                return per_token_logps,mini_batch_encoded_inputs['completion_mask']
        else:
            return per_token_logps

    def get_in_context_done(self,task:str,first_image:List[Image.Image],n_pre_image:List[Image.Image],now_image:List[Image.Image],ref_image_list:List[Image.Image],ref_num=9,rich=False):
        """
        In context 的方式获取done
        提供一段参考ref_image_list
        first_image是你的轨迹的第一张图,这里为list是可以batch推理
        n_pre_image是上一个时刻
        now_image是当前时刻
        """
        if ref_image_list is not None:
            ref_images=[ref_image_list[0]]
            delta=(len(ref_image_list)-1)/(ref_num-1)
            for i in range(1,ref_num):
                ref_images.append(ref_image_list[int(i*delta)])
        else:
            ref_num=0
        one_prompt=self.get_score_prompt(task=task,trajectory_len=ref_num,think=True)
        batch_prompt=[]
        batch_image=[]
        for i in range(len(now_image)):
            batch_prompt.append(one_prompt)
            batch_image.append(ref_images+[first_image[i],n_pre_image[i],now_image[i]])
        infer_requests=self.get_infer_requests(prompt=batch_prompt,images=batch_image)
        response_list,infer_time=self.chat(infer_requests)
        answers_list,complete_requests_list=self.results_format(response_list,infer_requests,rich=rich)
        think_pre_value_list=[]
        think_post_value_list=[]
        think_critic_list=[]
        for one in answers_list:
            one_critic=one.split('</think>')[1]
            one_pre_value=one.split('first image progressing: ')[1].split('%')[0]
            one_post_value=one.split('second image progressing: ')[1].split('%')[0]
            think_critic_list.append("{:.1f}".format((100-float(one_pre_value))*float(one_critic)/100.0))
            think_pre_value_list.append(one_pre_value)
            think_post_value_list.append("{:.3f}".format(float(one_post_value)/100.0))
        batch_done=think_post_value_list
        batch_related_critic=think_critic_list
        return batch_done,batch_related_critic

    def eval_trajectory(self, task,
                            batch_num=20, ref_num=9, skip=1, frame_skip=False, done_threshold=False, 
                            video_output=False, output_path=""):
        list_video = task.frames
        ref_list = task.ref_frames
        done_ref = ref_list
        critic_list, think_value_list=self.get_trajectory_critic(
            task=task.description,
            image_list=list_video,
            ref_image_list=ref_list,
            batch_num=batch_num,
            ref_num=ref_num,
            think=False,
            skip=skip,
            rich=False,
            reverse_eval=False,
            frame_skip=frame_skip,
            addition_scale=1,
            bias=0,
            positive_clip=0,
            negative_clip=0,
            related_critic=False,
        )
        value_list = think_value_list
        # if frame_skip:
        done_list = value_list
        if video_output:
            if ref_num>0:
                ref_image_paths=ref_list
            else:
                ref_image_paths=None
            fps = 1
            video_path = video_trajectory(
                traj_id=task.video_name[:-4],
                view='0',
                image_paths=[],
                image_objects=list_video,
                done_list=done_list,
                critic_list=critic_list,
                value_list=value_list,
                task=task.description,
                output_path=output_path,
                fps=fps,
                ref_image_paths=ref_image_paths,
                n_num=ref_num
            )
        
        return done_list
    
    def reward_step(
        self,
        pairs,
        pair_task_texts,
        *,
        use_ref: bool = True,
        batch_num: int = 256,
        temperature: float = 0.0,   # RM建议确定性
        top_k: int | None = None,   # None=贪心
        addition_scale: float = 1.0,
        divide_skip: int = 1,       # 如果上游做了frame skip，这里可把输出再 /skip
        value_simple: str | bool = "full",  # 透传给 critic_to_value_simple
        related_critic: bool = False,       # True则输出Δvalue代替原critic
        bias: float = 0.0,
        positive_clip: float = 0.0, # >0 则把(0, clip] 置为0，保留负值或大正值
        negative_clip: float = 0.0, # >0 则把[-clip,0) 置为0，保留正值或大负值
        return_value: bool = False, # 是否同时返回按序累计的value（仅当pairs来自同一条序列时有意义）
        rich: bool = False,         # 透传给 results_format
    ):
        
        assert len(pairs) == len(pair_task_texts), "pairs 与 pair_task_texts 长度必须一致"
        N = len(pairs)
        if N == 0:
            return ([], []) if return_value else []
        
        def _build_batch(start, end):
            batch_prompt = []
            batch_image = []
            for i in range(start, end):
                task = pair_task_texts[i]
                # 你的打分prompt接口：trajectory_len/think都走最简
                img_t   = pairs[i]["img_t"]
                img_tp1 = pairs[i]["img_tp1"]
                img_0 = pairs[i]["img_0"]
                img_ref = pairs[i]["img_ref"]
                
                one_prompt = self.get_score_prompt(task=task, trajectory_len=(len(img_ref) if use_ref else 0), think=False)
                batch_prompt.append(one_prompt)


                pair_imgs = img_ref + [img_0, img_t, img_tp1] if use_ref else [img_t, img_tp1]
                batch_image.append(pair_imgs)
            return batch_prompt, batch_image
        
        critic_raw = []
        
        self.temperature = temperature
        self.top_k = top_k
        
        # for s in tqdm.tqdm(range(0, N, batch_num), desc="RM batch inference"):
        for s in range(0, N, batch_num):
            e = min(s + batch_num, N)
            batch_prompt, batch_image = _build_batch(s, e)
            infer_requests = self.get_infer_requests_client(prompts=batch_prompt, images=batch_image)
            response_list, infer_time = self.chat(infer_requests)
            answers_list, complete_requests_list = self.results_format(
                response_list, infer_requests, rich=rich
            )
            def _to_float(x):
                if isinstance(x, (int, float)):
                    return float(x)
                xs = str(x).strip()
                if xs.endswith("%"):
                    xs = xs[:-1]
                try:
                    return float(xs)
                except Exception:
                    # 兜底：如果无法解析，置0并打印警告
                    # 你也可以选择 raise
                    print(f"[RM warn] progress parse failed for: {x}, set to 0.0")
                    return 0.0

            critic_raw.extend([_to_float(a) for a in answers_list])
        
        critic_list = [float(c) for c in critic_raw]

        if divide_skip and divide_skip != 1:
            critic_list = [c / float(divide_skip) for c in critic_list]

        if addition_scale and addition_scale != 1.0:
            critic_list = [c / float(addition_scale) for c in critic_list]

        if bias != 0.0:
            critic_list = [c + float(bias) for c in critic_list]

        if positive_clip and positive_clip > 0.0:
            # 把 (0, positive_clip] 置0，保留负值或大正值
            critic_list = [c if (c < 0 or c > positive_clip) else 0.0 for c in critic_list]

        if negative_clip and negative_clip > 0.0:
            # 把 [-negative_clip, 0) 置0，保留正值或大负值
            critic_list = [c if (c > 0 or c < -negative_clip) else 0.0 for c in critic_list]

        # --- 4) 需要的话，折算为 value 或 Δvalue ---
        if related_critic:
            # 用你的累计函数得到 value，再取 Δvalue（即 Δvalue 替代 critic）
            value_list = self.critic_to_value_simple(critic_list, simple=value_simple)
            # value_list 长度 = len(critic_list)+1
            critic_related = [value_list[i] - value_list[i - 1] for i in range(1, len(value_list))]
            if return_value:
                return critic_related, value_list
            else:
                return critic_related
        else:
            if return_value:
                value_list = self.critic_to_value_simple(critic_list, simple=True)
                return critic_list, value_list
            else:
                return critic_list
        
            
    def web_trajectory_critic(self, task_description, main_video_path, reference_video_path=None, 
                            batch_num=20, ref_num=9, think=False, skip=1, rich=False, reverse_eval=False,output_path=None,fps=None,frame_skip=False,addition_scale=1,bias=0,positive_clip=0,negative_clip=0,related_critic=False,done_flag=False,in_context_done=False,done_threshold=False,video_output=True):
        list_video=images_get_from_video(main_video_path)
        ref_list=None
        if reference_video_path is None:
            ref_num=0
        else:
            ref_list=images_get_from_video(reference_video_path)
        if done_flag:
            if in_context_done:
                done_ref=ref_list
            else:
                done_ref=None
            done_list=self.get_trajectory_done(task=task_description,image_list=list_video,ref_image_list=done_ref,batch_num=batch_num,rich=True,ref_num=ref_num,threshold=done_threshold)
            if frame_skip:
                done_list=[done_list[i] for i in range(0,len(done_list),skip)]
        else:
            done_list=None
        critic_list, think_value_list=self.get_trajectory_critic(task=task_description,image_list=list_video,ref_image_list=ref_list,batch_num=batch_num,ref_num=ref_num,think=think,skip=skip,rich=rich,reverse_eval=reverse_eval,frame_skip=frame_skip,addition_scale=addition_scale,bias=bias,positive_clip=positive_clip,negative_clip=negative_clip,related_critic=related_critic)
        value_list=think_value_list
        if video_output:
            if ref_num>0:
                ref_image_paths=ref_list
            else:
                ref_image_paths=None
            if fps:
                pass
            else:
                fps=5.0/skip
            video_path = video_trajectory(
                traj_id='temp',
                view='0',
                image_paths=[],
                image_objects=list_video,
                done_list=done_list,
                critic_list=critic_list,
                value_list=value_list,
                task=task_description,
                output_path=output_path,
                fps=fps,
                ref_image_paths=ref_image_paths,
                n_num=ref_num
            )
        else:
            video_path=None
        return video_path,value_list,critic_list,done_list
    
    def web_trajectory_done(self, task_description, main_image_path, reference_image_path=None, rich=False):
        done_list=self.get_trajectory_done(task=task_description,image_list=[main_image_path],batch_num=1,rich=rich)
        return f"Task: {task_description}\nCritic Score: {done_list[0]}"
    
    def get_trajectory_task(self, image_list):
        one_prompt=self.get_task_prompt()
        infer_requests=self.get_infer_requests(prompt=one_prompt,images=[image_list[0],image_list[-1]])
        response_list,infer_time=self.chat(infer_requests)
        answers_list,complete_requests_list=self.results_format(response_list,infer_requests)
        return answers_list

    def get_trajectory_done(self,task:str,image_list:List[Image.Image],ref_image_list:List[Image.Image]=None,batch_num:int=20,rich=False,ref_num=9,threshold=0,skip=1,goal_image:Image.Image=None):
        """
        输入一条trajectory的所有图片,输出每张图片的done,0~1的突变值
        当有ref_image_list时,进行过程判断,输出0~1的渐变值
        当有gaol_image时,进行最终状态判断,输出0~1的突变值(与当有ref_image_list冲突),task=None时只依靠图片
        """
        batch_prompt=[]
        batch_image=[]
        done_list=[]
        if skip>1:
            image_list=[image_list[i] for i in range(0,len(image_list),skip)]
        if ref_image_list is None and goal_image is None:
            for i in tqdm.tqdm(range(len(image_list)),desc='done processing'):
                one_prompt=self.get_done_prompt(task=task)
                batch_prompt.append(one_prompt)
                batch_image.append([image_list[i]])
                if (i+1) % batch_num==0 or i==len(image_list)-1:
                    infer_requests=self.get_infer_requests(prompt=batch_prompt,images=batch_image)
                    response_list,infer_time=self.chat(infer_requests)
                    answers_list,complete_requests_list=self.results_format(response_list,infer_requests,rich=rich)
                    print(f'infer_time:{infer_time}s')
                    print(f'answers_list:{answers_list}')
                    done_list.extend(answers_list)
                    batch_prompt=[]
                    batch_image=[]
        elif ref_image_list:
            first_image=[]
            n_pre_image=[]
            now_image=[]
            for i in tqdm.tqdm(range(1,len(image_list)),desc='done processing'):
                first_image.append(image_list[0])
                n_pre_image.append(image_list[i-1])
                now_image.append(image_list[i])
                if (i+1) % batch_num==0 or i==len(image_list)-1:
                    batch_done,batch_related_critic=self.get_in_context_done(task=task,first_image=first_image,n_pre_image=n_pre_image,now_image=now_image,ref_image_list=ref_image_list,ref_num=ref_num,rich=rich)
                    print(f'answers_list:{batch_done}')
                    done_list.extend(batch_done)
                    first_image=[]
                    n_pre_image=[]
                    now_image=[]
            done_list=[0]+done_list
        else:
            for i in tqdm.tqdm(range(len(image_list)),desc='done processing'):
                one_prompt=self.get_in_context_done_prompt(task=task)
                batch_prompt.append(one_prompt)
                batch_image.append([goal_image,image_list[i]])
                if (i+1) % batch_num==0 or i==len(image_list)-1:
                    infer_requests=self.get_infer_requests(prompt=batch_prompt,images=batch_image)
                    response_list,infer_time=self.chat(infer_requests)
                    answers_list,complete_requests_list=self.results_format(response_list,infer_requests,rich=rich)
                    print(f'infer_time:{infer_time}s')
                    print(f'answers_list:{answers_list}')
                    done_list.extend(answers_list)
                    batch_prompt=[]
                    batch_image=[]
        done_list = [float(one) if float(one) > threshold else 0.0 for one in done_list]
        return done_list

    def get_trajectory_critic(self,task:str,image_list:List[Image.Image],ref_image_list:List[Image.Image]=None,batch_num:int=20,ref_num=9,think=False,skip=1,rich=False,reverse_eval=False,frame_skip=True,addition_scale=1,bias=0,related_critic=False,positive_clip=0,negative_clip=0,value_simple=True):
        """
        输入一条trajectory的所有图片,输出每张图片的critic和processing value
        可以给一条参考轨迹
        """
        batch_prompt=[]
        batch_image=[]
        critic_list=[]
        value_list=[]
        if ref_image_list is not None:
            ref_images=[ref_image_list[0]]
            delta=(len(ref_image_list)-1)/(ref_num-1)
            for i in range(1,ref_num):
                ref_images.append(ref_image_list[int(i*delta)])
        else:
            ref_num=0
        if frame_skip:
            select_idx=range(skip,len(image_list),skip)
        else:
            select_idx=range(skip,len(image_list))
        for i in tqdm.tqdm(select_idx,desc='critic processing'):
            one_prompt=self.get_score_prompt(task=task,trajectory_len=ref_num,think=think)
            batch_prompt.append(one_prompt)
            if ref_image_list is not None:
                if reverse_eval:
                    batch_image.append(ref_images+[image_list[0],image_list[i],image_list[i-skip]])
                else:
                    batch_image.append(ref_images+[image_list[0],image_list[i-skip],image_list[i]])
            else:
                if reverse_eval:
                    batch_image.append([image_list[i],image_list[i-skip]])
                else:
                    batch_image.append([image_list[i-skip],image_list[i]])
            if (len(batch_prompt)) % batch_num==0 or len(critic_list)+len(batch_prompt)==len(select_idx):
                infer_requests=self.get_infer_requests(prompt=batch_prompt,images=batch_image)
                response_list,infer_time=self.chat(infer_requests)
                answers_list,complete_requests_list=self.results_format(response_list,infer_requests,rich=rich)
                print(f'infer_time:{infer_time}s')
                print(f'answers_list:{answers_list}')
                critic_list.extend(answers_list)
                batch_prompt=[]
                batch_image=[]
        if think:
            think_pre_value_list=[]
            think_post_value_list=[]
            think_critic_list=[]
            for one in critic_list:
                one_critic=one.split('</think>')[1]
                one_pre_value=one.split('first image progressing: ')[1].split('%')[0]
                one_post_value=one.split('second image progressing: ')[1].split('%')[0]
                think_critic_list.append(one_critic)
                think_pre_value_list.append(one_pre_value)
                think_post_value_list.append(one_post_value)
            if reverse_eval:
                temp=think_pre_value_list
                think_pre_value_list=think_post_value_list
                think_post_value_list=temp
                think_critic_list=[0-float(one) for one in think_critic_list]
            think_pre_value_list=think_pre_value_list+think_post_value_list[-skip:]
            think_post_value_list=think_pre_value_list[:skip]+think_post_value_list
            critic_list=think_critic_list
            if frame_skip:
                pass
            else:
                critic_list=[float(one)/skip for one in critic_list]
            critic_list=[float(one)/addition_scale for one in critic_list]
            value_list=self.critic_to_value_simple(critic_list,simple=value_simple)
        else:
            if reverse_eval:
                critic_list=[0-float(one) for one in critic_list]
            if frame_skip:
                pass
            else:
                critic_list=[float(one)/skip for one in critic_list]
            critic_list=[float(one)/addition_scale for one in critic_list]
            value_list=self.critic_to_value_simple(critic_list,simple=value_simple)
        if related_critic:
            critic_list=[value_list[i]-value_list[i-1] for i in range(1,len(value_list))]
        if bias!=0:
            critic_list=[one+bias for one in critic_list]
        if positive_clip!=0:
            critic_list=[one if (one<0 or one>positive_clip) else 0 for one in critic_list]
        if negative_clip!=0:
            critic_list=[one if (one>0 or one<-negative_clip) else 0 for one in critic_list]
        if related_critic:
            value_list=[0]
            for one in critic_list:
                value_list.append(value_list[-1]+one)
        else:
            value_list=self.critic_to_value_simple(critic_list,simple=value_simple)
        #这里的value也是done，越接近100完成度越高
        return critic_list,value_list

    def magic_smooth(self,task,file_path,hz=10,value_simple='mix_f',ref_image_list=None,ref_num=9,think=False,max_skip=3):
        import os
        import pickle
        import gzip
        import subprocess
        from video_tool import read_data_and_create_video,images_get_from_video
        if type(ref_image_list)  == str:
            if ref_image_list.endswith(".pkl.gz"):
                ref_image_list = read_data_and_create_video(ref_image_list)
            else:
                ref_image_list=images_get_from_video(ref_image_list)
        image_list=read_data_and_create_video(file_path, camera_name=None, create_video=False, output_path=None, fps=hz, start_frame=0, end_frame=None)
        skip=int(hz/2)
        critic_list,value_list=self.get_trajectory_critic(task=task,image_list=image_list,ref_image_list=ref_image_list,batch_num=10,ref_num=ref_num,think=think,skip=skip,rich=True,reverse_eval=False,frame_skip=True,addition_scale=1,bias=0,related_critic=False,positive_clip=0,negative_clip=0,value_simple=value_simple)

        skip_idx_range=[]
        skip_idxs=[]
        i=0
        while i <len(critic_list):
            if float(critic_list[i])<=0:
                skip_window=2
                while i+skip_window<len(critic_list) and skip_window<=max_skip:
                    two_image=[image_list[0],image_list[i*skip],image_list[(i+skip_window)*skip]]
                    one_critic_list,_=self.get_trajectory_critic(task=task,image_list=two_image,ref_image_list=ref_image_list,batch_num=10,ref_num=ref_num,think=think,skip=1,rich=True,reverse_eval=False,frame_skip=True,addition_scale=1,bias=0,related_critic=False,positive_clip=0,negative_clip=0,value_simple=value_simple)
                    if float(one_critic_list[-1])<0:
                        skip_window+=1
                    else:
                        break
                skip_idx_range.append([i,i+skip_window])
                skip_idxs.extend(range(i*skip+1,(i+skip_window)*skip))
                i+=skip_window
                print("skip_window:",skip_window)
            else:
                i+=1
        with gzip.open(file_path, 'rb') as f:
            original_data = pickle.load(f)
        
        base_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0]  # 去掉.pkl.gz
        output_dir = os.path.join(base_dir, f"{file_name}_smoothed")
        os.makedirs(output_dir, exist_ok=True)
        new_data=[one for i,one in enumerate(original_data) if i not in skip_idxs]
        new_images=[one for i,one in enumerate(image_list) if i not in skip_idxs]

        data_file = os.path.join(output_dir, f"{file_name}_smoothed.pkl.gz")
            
        with gzip.open(data_file, 'wb') as f:
            pickle.dump(new_data, f)
        
        video_file = os.path.join(output_dir, f"{file_name}_smoothed.mp4")
        
        temp_dir = os.path.join(output_dir, f"temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            for j, img in enumerate(new_images):
                frame_file = os.path.join(temp_dir, f"frame_{j:06d}.png")
                if hasattr(img, 'save'):
                    img.save(frame_file)
                else:
                    from PIL import Image
                    if isinstance(img, np.ndarray):
                        Image.fromarray(img).save(frame_file)
                    else:
                        Image.fromarray(np.array(img)).save(frame_file)
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(hz),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                video_file
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"Created video: {video_file}")
            
        except Exception as e:
            print(f"Error creating video for segment {i}: {e}")
            
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        return new_data
        
    def magic_split(self,task,file_path,hz=10,value_simple='mix_f',done_assist=False,ref_image_list=None):
        import os
        import pickle
        import gzip
        import subprocess
        from video_tool import read_data_and_create_video,images_get_from_video
        from magic_detect import adaptive_peak_valley_detection,plot_results
        if type(ref_image_list)  == str:
            if ref_image_list.endswith(".pkl.gz"):
                ref_image_list = read_data_and_create_video(ref_image_list)
            else:
                ref_image_list=images_get_from_video(ref_image_list)
        
        skip=int(hz/2)
        done_len=4
        gap_merge_len=4#一般看数据的平均记录周期
        image_list=read_data_and_create_video(file_path, camera_name=None, create_video=False, output_path=None, fps=hz, start_frame=0, end_frame=None)
        critic_list,value_list=self.get_trajectory_critic(task=task,image_list=image_list,ref_image_list=ref_image_list,batch_num=10,ref_num=9,think=False,skip=skip,rich=True,reverse_eval=False,frame_skip=True,addition_scale=1,bias=0,related_critic=False,positive_clip=0,negative_clip=0,value_simple=value_simple)
        
        if done_assist:
            done_list=self.get_trajectory_done(task=task,image_list=image_list,ref_image_list=None,batch_num=10,ref_num=2,rich=True,threshold=0.9,skip=skip)
            assist_peaks={}
            peak_window=[]
            done_list=done_list+[0]
            for k in range(len(done_list)):
                if done_list[k]>=0.95:
                    peak_window.append(k)
                else:
                    if len(peak_window)>=done_len:
                        assist_peaks[peak_window[done_len-1]]=peak_window
                    peak_window=[]
            peak_list=list(assist_peaks.keys())
            index=0
            while index<len(peak_list)-1:
                if assist_peaks[peak_list[index+1]][0]-assist_peaks[peak_list[index]][-1]<=gap_merge_len:
                    assist_peaks[peak_list[index]].extend(assist_peaks[peak_list[index+1]])
                    del assist_peaks[peak_list[index+1]]
                    del peak_list[index+1]
                else:
                    index+=1
            result = adaptive_peak_valley_detection(
                        value_list, 
                        assist_peaks=peak_list,
                        window_size=None,  # Adaptive window size
                        verbose=True,
                        min_distance=10,
                        prominence_threshold=0.1,
                        outlier_sensitivity=3
            )
        else:
            result = adaptive_peak_valley_detection(
                        value_list, 
                        window_size=None,  # Adaptive window size
                        verbose=True,
                        min_distance=10,
                        prominence_threshold=0.01,
                        outlier_sensitivity=1.5
            )
        

        with gzip.open(file_path, 'rb') as f:
            original_data = pickle.load(f)
        
        base_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0]  # 去掉.pkl.gz
        output_dir = os.path.join(base_dir, f"{file_name}_segments")
        os.makedirs(output_dir, exist_ok=True)
        plot_results(value_list, result,save_path=os.path.join(output_dir, "peak_valley.png"))
        peaks = [one*skip for one in result['peaks']]
        valleys = [one*skip for one in result['valleys']]
        
        ascending_dir = os.path.join(output_dir, "ascending")
        descending_dir = os.path.join(output_dir, "descending")
        os.makedirs(ascending_dir, exist_ok=True)
        os.makedirs(descending_dir, exist_ok=True)
        
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        all_points = []
        for p in peaks:
            all_points.append((p, 'peak'))
        for v in valleys:
            all_points.append((v, 'valley'))
        all_points.sort(key=lambda x: x[0])
        
        segments = []
        
        for i, (point_idx, point_type) in enumerate(all_points):
            if i == 0:
                prev_type='valley'
                start_idx = 0
            else:
                start_idx=all_points[i-1][0]
                prev_type = all_points[i-1][1]

            end_idx = point_idx
            if prev_type == 'valley' and point_type == 'peak':
                segment_type = 'ascending'
            elif prev_type == 'peak' and point_type == 'valley':
                segment_type = 'descending'
            else:
                continue
                
            data_segment = original_data[start_idx:end_idx]
            image_segment = image_list[start_idx:end_idx]
            
            segments.append({
                'type': segment_type,
                'start': start_idx,
                'end': end_idx,
                'data': data_segment,
                'images': image_segment
            })
            
        
        if segments and segments[-1]['type'] == 'descending':
            segments.pop()
        
        for i, segment in enumerate(segments):
            segment_type = segment['type']
            
            task_name = self.get_trajectory_task(segment['images'])[0].split('</task>')[0].strip().replace(' ', '_').replace('.', '_').replace(',', '_')
            
            if segment_type == 'ascending':
                data_file = os.path.join(ascending_dir, f"{task_name}_{i:03d}.pkl.gz")
            else:
                data_file = os.path.join(descending_dir, f"{task_name}_{i:03d}.pkl.gz")
                
            with gzip.open(data_file, 'wb') as f:
                pickle.dump(segment['data'], f)
            
            video_file = os.path.join(video_dir, f"{segment_type}_{task_name}_{i:03d}.mp4")
            
            temp_dir = os.path.join(output_dir, f"temp_{i}")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                for j, img in enumerate(segment['images']):
                    frame_file = os.path.join(temp_dir, f"frame_{j:06d}.png")
                    if hasattr(img, 'save'):
                        img.save(frame_file)
                    else:
                        from PIL import Image
                        if isinstance(img, np.ndarray):
                            Image.fromarray(img).save(frame_file)
                        else:
                            Image.fromarray(np.array(img)).save(frame_file)
                
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(hz),
                    '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    video_file
                ]
                
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Created video: {video_file}")
                
            except Exception as e:
                print(f"Error creating video for segment {i}: {e}")
                
            finally:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        print(f"Processing complete. Segments saved to: {output_dir}")
        print(f"Total segments: {len(segments)}")
        print(f"Ascending segments: {sum(1 for s in segments if s['type'] == 'ascending')}")
        print(f"Descending segments: {sum(1 for s in segments if s['type'] == 'descending')}")
        
        return segments
    
    def critic_to_value_simple(self,critic_list,simple=True):
        """
        将critic计算为0-100的value,输入需是一条轨迹按顺序的critic
        """
        value_list=[0]
        for i in range(len(critic_list)):
            if float(critic_list[i])>0 or simple is True:
                value_list.append(value_list[-1]+(100-value_list[-1])*float(critic_list[i])/100.0)
            else:
                if simple=='mix_f':
                    if value_list[-1]>50:
                        value_list.append(max(10,100-max((100-value_list[-1]),1.0)/(100+float(critic_list[i]))*100.0))
                    else:
                        value_list.append(value_list[-1]+value_list[-1]*float(critic_list[i])/100.0)
                else:
                    value_list.append(100-max((100-value_list[-1]),1.0)/(100+float(critic_list[i]))*100.0)
        return value_list
    
    def compute_voc(self,values_list):
        """
        计算Value-Order Correlation (VOC)
        
        参数:
            predicted_values: 模型预测的值序列，形状为(T,)，其中T是时间步数
            
        返回:
            voc: Value-Order Correlation值，范围从-1到1
        """
        T = len(values_list)
        time_order = np.arange(T)
        
        correlation, _ = spearmanr(values_list, time_order)
        return correlation
    
    def compute_negative_rate(self,critic_list):
        """
        计算逆序动作的比例
        """
        negative_critic=[one for one in critic_list if one<0]

        return float(len(negative_critic))/len(critic_list)


def import_external_file(file_path: str):
    import importlib
    file_path = os.path.abspath(os.path.expanduser(file_path))
    py_dir, py_file = os.path.split(file_path)
    assert os.path.isdir(py_dir), f'py_dir: {py_dir}'
    sys.path.insert(0, py_dir)
    return importlib.import_module(py_file.split('.', 1)[0])