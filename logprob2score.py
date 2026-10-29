import math
import json
import argparse
from tqdm import tqdm

def safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0

def token_contains_real(token: str) -> bool:
    return "real" in token.lower()

def token_contains_fake(token: str) -> bool:
    return "fake" in token.lower()

def normalize_binary_prob(p0: float, p1: float):
    s = p0 + p1
    if s <= 0:
        return 0.5, 0.5
    return p0 / s, p1 / s

def logprobs_to_probs(pred_jsonl_line) -> tuple:
    """
    解析单行预测结果，返回 (p_fake, p_real)
    """
    if isinstance(pred_jsonl_line, str):
        pred_jsonl_line = json.loads(pred_jsonl_line)
    
    # 兼容 logprobs 字段可能直接是 content 列表，或者是包含 content 的字典
    logprobs_data = pred_jsonl_line.get('logprobs', {})
    if isinstance(logprobs_data, dict):
        content = logprobs_data.get('content', [])
    elif isinstance(logprobs_data, list):
        content = logprobs_data
    else:
        content = []

    if not content:
        return 0.5, 0.5

    first_token_info = content[0]
    top_logprobs = first_token_info.get("top_logprobs", [])

    p_real = 0.0
    p_fake = 0.0
    top20_sum = 0.0

    for cand in top_logprobs[:20]:
        token = cand.get("token", "")
        logprob = cand.get("logprob", float("-inf"))
        prob = safe_exp(logprob)
        top20_sum += prob

        if token_contains_real(token):
            p_real += prob
        if token_contains_fake(token):
            p_fake += prob

    top20_sum = min(max(top20_sum, 0.0), 1.0)

    has_real = p_real > 0
    has_fake = p_fake > 0

    if has_real and has_fake:
        _, p_fake = normalize_binary_prob(p_real, p_fake)
    elif has_real and not has_fake:
        p_fake_est = max(0.0, 1.0 - top20_sum)
        _, p_fake = normalize_binary_prob(p_real, p_fake_est)
    elif has_fake and not has_real:
        p_real_est = max(0.0, 1.0 - top20_sum)
        _, p_fake = normalize_binary_prob(p_real_est, p_fake)
    else:
        p_fake = 0.5

    p_real_final = 1.0 - p_fake
    return p_fake, p_real_final

def extract_file_path(data: dict) -> str:
    """
    从数据中提取文件路径，兼容 images 和 videos 字段
    """
    if "images" in data and data["images"]:
        return data["images"][0]
    elif "videos" in data and data["videos"]:
        return data["videos"][0]
    return ""

def process_file(input_path: str, output_path: str):
    """
    读取输入jsonl文件，计算概率，并写入输出jsonl文件
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        # 先统计总行数用于 tqdm 进度条
        total_lines = sum(1 for _ in fin)
        fin.seek(0)  # 重置文件指针到开头

        for line in tqdm(fin, total=total_lines, desc="Processing"):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            file_path = extract_file_path(data)
            p_fake, p_real = logprobs_to_probs(data)

            # 构造输出字典
            out_data = {
                "file_path": file_path,
                "p_fake": round(p_fake, 6),
                "p_real": round(p_real, 6)
            }

            # ensure_ascii=False 保证中文路径不被转义
            fout.write(json.dumps(out_data, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Convert ms-swift prediction jsonl to binary classification probabilities.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input prediction jsonl file.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output probability jsonl file.")
    
    args = parser.parse_args()
    
    process_file(args.input, args.output)
    print(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()
