"""
Multi-request and multi-concurrency Tool

Usage:
    python test_perf.py <num_requests> <concurrency> <api_url> <model> [--verbose]

Arguments:
    <num_requests>    Number of total requests to send (e.g. 100)
    <concurrency>     Number of concurrent requests (e.g. 10)
    <api_url>         OpenAI-compatible base URL (e.g. http://localhost:8000)
    <model>           Model name to test (e.g. jiuge)
    --verbose         (Optional) Print per-request details

Example:
    python scripts/test_perf.py 50 5 http://127.0.0.1:8000 jiuge --verbose
"""

import argparse
import asyncio
import time
import random
from openai import AsyncOpenAI

PROMPTS = [
    "如果猫能写诗，它们会写些什么？",
    "描述一个没有重力的世界。",
    "如果地球停止自转，会发生什么？",
    "假设你是一只会飞的鲸鱼，描述你的日常生活。",
    "如果人类可以与植物沟通，世界会变成什么样？",
    "描述一个由糖果构成的城市。",
    "如果时间旅行成为可能，你最想去哪个时代？",
    "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
    "如果动物能上网，它们会浏览什么网站？",
    "描述一个没有声音的世界。",
    "如果人类可以在水下呼吸，城市会如何变化？",
    "想象一下，如果天空是绿色的，云是紫色的。",
    "如果你能与任何历史人物共进晚餐，你会选择谁？",
    "描述一个没有夜晚的星球。",
    "如果地球上只有一种语言，世界会如何运作？",
    "想象一下，如果所有的书都变成了音乐。",
    "如果你可以变成任何一种动物，你会选择什么？",
    "描述一个由机器人统治的未来世界。",
    "如果你能与任何虚构角色成为朋友，你会选择谁？",
    "想象一下，如果每个人都能读懂他人的思想。"
]

def ljust_cn(s: str, width: int) -> str:
    """Pad a string to width accounting for mixed char widths"""
    # crude approximation: Chinese chars count as width 2
    pad = width - sum(2 if ord(c) > 255 else 1 for c in s)
    return s + " " * max(0, pad)

async def benchmark_user(worker_id, client, semaphore, queue, results, model, verbose):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break
            start = time.time()
            prompt = random.choice(PROMPTS)
            if verbose:
                print(f"🚀 [Req {task_id}] Worker-{worker_id} Prompt: {prompt}")
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                first_time = None
                tokens = 0
                response = ""
                async for chunk in stream:
                    txt = chunk.choices[0].delta.content
                    if txt:
                        response += txt
                        tokens += 1
                        if first_time is None:
                            first_time = time.time()
                    if chunk.choices[0].finish_reason is not None:
                        break
                end = time.time()
                latency = end - start
                ttft = (first_time - start) if first_time else 0
                avg_tok = latency / tokens * 1000 if tokens else 0
                results.append((task_id, worker_id, ttft, latency, tokens, avg_tok, prompt, response))
                if verbose:
                    print("""
   TTFT            : {:.3f} s
   Latency         : {:.3f} s
   Tokens         : {}
   Avg/token      : {:.2f} ms
   Response       : {}...
""".format(ttft, latency, tokens, avg_tok, response[:100]))
            except Exception as e:
                print(f"❌ [Req {task_id}] Worker-{worker_id} failed: {e}")
            finally:
                queue.task_done()

async def run_benchmark(num_requests, concurrency, api_url, model, verbose):
    print(f"🔗 Connecting to {api_url}, model {model}")
    client = AsyncOpenAI(base_url=api_url, api_key="default")
    sem = asyncio.Semaphore(concurrency)
    queue = asyncio.Queue()
    results = []
    for i in range(num_requests): await queue.put(i)
    for _ in range(concurrency): await queue.put(None)
    tasks = [asyncio.create_task(benchmark_user(i, client, sem, queue, results, model, verbose)) for i in range(concurrency)]
    st = time.time()
    await queue.join()
    await asyncio.gather(*tasks)
    et = time.time()
    total = et - st
    success = len(results)
    rps = success / total if total else 0
    total_tokens = sum(r[4] for r in results)
    avg_latency = sum(r[3] for r in results)/success if success else 0
    avg_ttft = sum(r[2] for r in results)/success if success else 0
    avg_tok_ms = sum(r[5] for r in results)/success if success else 0
    tok_speed = total_tokens/total if total else 0
    w = 26
    print("\n=== 📊 性能指标汇总 ===")
    print("-"*60)
    print(f"{ljust_cn('并发数',w)}: {concurrency}")
    print(f"{ljust_cn('请求总数',w)}: {num_requests}")
    print(f"{ljust_cn('成功请求数',w)}: {success}")
    print(f"{ljust_cn('总耗时',w)}: {total:>7.2f} s")
    print(f"{ljust_cn('总输出token数',w)}: {total_tokens}")
    print(f"{ljust_cn('请求速率 (RPS)',w)}: {rps:>7.2f} req/s")
    print(f"{ljust_cn('Average latency',w)}: {avg_latency:>7.2f} s")
    print(f"{ljust_cn('Average TTFT',w)}: {avg_ttft:>7.2f} s")
    print(f"{ljust_cn('Avg time per token',w)}: {avg_tok_ms:>7.2f} ms/token")
    print(f"{ljust_cn('Token generation speed',w)}: {tok_speed:>7.2f} tok/s")
    print("-"*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-concurrency performance testing')
    parser.add_argument('num_requests', type=int, help='total requests')
    parser.add_argument('concurrency', type=int, help='parallel workers')
    parser.add_argument('api_url', type=str, help='API base URL')
    parser.add_argument('model', type=str, help='model name')
    parser.add_argument('--verbose', action='store_true', help='detailed per-request info')
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.num_requests, args.concurrency, args.api_url, args.model, args.verbose))
