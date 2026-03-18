import os
import json
import base64
import torch
import numpy as np
from PIL import Image
from openai import OpenAI

# ==========================================
# 1. 图像预处理 (转换为 Base64 供 API 使用)
# ==========================================
def encode_image_to_base64(image_path, target_size=(224, 224)):
    """
    将参考图和目标图统一缩放至 224x224，并转换为 Base64 格式
    以便通过多模态 API 发送给 GPT-4o。
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(target_size)
        # 临时保存以获取 base64
        import io
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

# ==========================================
# 2. 核心 API 调用机制 (Zero-Shot Structured Prompting)
# ==========================================
class GPT4o_SpatialEvaluator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        # 图像标准输出尺寸
        self.img_size = 224

    def evaluate_change(self, ref_img_path, tar_img_path, intent_text):
        """
        论文核心：通过少样本/零样本提示，要求模型输出归一化的坐标 JSON 数组。
        """
        ref_base64 = encode_image_to_base64(ref_img_path)
        tar_base64 = encode_image_to_base64(tar_img_path)
        
        # 构建严格的结构化 Prompt (对应论文: Zero-shot structured prompting)
        system_prompt = (
            "You are a strict and precise GUI change detection system. "
            "You will be provided with a reference UI screenshot, a target UI screenshot, and a natural language description of the change intent. "
            "Your ONLY task is to identify the spatial regions (bounding boxes) that changed according to the intent. "
            "You MUST output the result as a strictly formatted JSON array containing normalized coordinates [x_min, y_min, x_max, y_max] where values are floats between 0.0 and 1.0. "
            "Do not include any other text or explanation. Output format: [[0.1, 0.2, 0.3, 0.4], ...]"
        )
        
        user_prompt = f"Change Intent: {intent_text}"

        try:
            # 论文中明确指出：Temperature 设为 0.1 以确保输出结构的确定性
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": ref_base64}},
                            {"type": "image_url", "image_url": {"url": tar_base64}}
                        ]
                    }
                ],
                temperature=0.1, 
                max_tokens=300,
                response_format={ "type": "json_object" } # 强制 JSON 输出
            )
            
            raw_output = response.choices[0].message.content
            # 解析 GPT-4o 返回的 JSON 数组
            parsed_coords = json.loads(raw_output)
            
            # 兼容可能的 JSON 嵌套结构提取
            if isinstance(parsed_coords, dict):
                # 假设模型返回 {"boxes": [[...]]}
                parsed_coords = list(parsed_coords.values())[0]
                
            return parsed_coords
            
        except Exception as e:
            print(f"API Call or Parsing failed: {e}")
            return []

# ==========================================
# 3. 空间渲染逻辑 (将归一化坐标渲染为二值掩码)
# ==========================================
def render_gpt_mask(normalized_boxes, img_size=224):
    """
    论文核心后处理：Parsed coordinates are scaled and rendered as filled rectangles on a binary mask.
    将 GPT-4o 输出的抽象浮点数坐标，转换为 224x224 的像素掩码。
    """
    # 初始化全 0 的二值掩码 (H, W)
    mask = torch.zeros((img_size, img_size), dtype=torch.float32)
    
    if not normalized_boxes:
        return mask
        
    for box in normalized_boxes:
        # 解包归一化坐标 [0.0 ~ 1.0]
        x_min_norm, y_min_norm, x_max_norm, y_max_norm = box
        
        # 缩放至 224x224 像素空间并取整 (Scaled to approximate localization)
        x_min = int(max(0, x_min_norm * img_size))
        y_min = int(max(0, y_min_norm * img_size))
        x_max = int(min(img_size, x_max_norm * img_size))
        y_max = int(min(img_size, y_max_norm * img_size))
        
        # 将被识别为发生变化的矩形区域填充为 1
        if x_max > x_min and y_max > y_min:
            mask[y_min:y_max, x_min:x_max] = 1.0
            
    # 扩展维度为 (1, 224, 224) 以匹配评估脚本的格式
    return mask.unsqueeze(0)

# ==========================================
# 4. 执行完整的评估流程
# ==========================================
def run_gpt4o_evaluation():
    api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    evaluator = GPT4o_SpatialEvaluator(api_key=api_key)
    
    # 模拟从测试集加载的一组数据
    ref_image = "path/to/ref.jpg"
    tar_image = "path/to/tar.jpg"
    intent = "Moved the 'Submit' button downwards by 10 pixels."
    
    # 1. 调用大模型获取归一化坐标
    print("Calling GPT-4o for spatial reasoning...")
    boxes = evaluator.evaluate_change(ref_image, tar_image, intent)
    print(f"GPT-4o predicted normalized boxes: {boxes}")
    
    # 2. 渲染为评估所需的像素级掩码
    predicted_mask = render_gpt_mask(boxes, img_size=224)
    print(f"Rendered Mask Shape: {predicted_mask.shape}")
    print(f"Number of activated pixels: {predicted_mask.sum().item()}")
    
    # 接下来即可将 predicted_mask 与 ground_truth_mask 计算 F1, IoU, Accuracy...

if __name__ == '__main__':
    # run_gpt4o_evaluation()
    pass
