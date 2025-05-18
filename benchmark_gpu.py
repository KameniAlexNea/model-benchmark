import time
import torch
import pynvml
import argparse
import random # Added import
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fvcore.nn import FlopCountAnalysis, parameter_count


def get_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return mem.used / (1024 ** 2)


DIVERSE_SENTENCES = [
    "Hello, how are you?",
    "This is a longer sentence to test the model's performance with more tokens.",
    "The quick brown fox jumps over the lazy dog, and other fables from ancient times often contain moral lessons.",
    "Bonjour, comment ça va ?",
    "Ceci est une phrase plus longue pour tester les performances du modèle avec plus de jetons.",
    "Le renard brun et rapide saute par-dessus le chien paresseux, et d'autres fables des temps anciens contiennent souvent des leçons de morale.",
    "Hola, ¿cómo estás?",
    "Esta es una frase más larga para probar el rendimiento del modelo con más tokens.",
    "El veloz zorro marrón salta sobre el perro perezoso, y otras fábulas de la antigüedad suelen contener lecciones morales.",
    "Hallo, wie geht es dir?",
    "Dies ist ein längerer Satz, um die Leistung des Modells mit mehr Token zu testen.",
    "Der schnelle braune Fuchs springt über den faulen Hund, und andere Fabeln aus alten Zeiten enthalten oft moralische Lehren.",
    "你好，你好吗？",
    "这是一个较长的句子，用于测试模型在处理更多标记时的性能。",
    "敏捷的棕色狐狸跳过了懒狗，古代的其他寓言通常包含道德教训。",
    "こんにちは、お元気ですか？",
    "これは、より多くのトークンでモデルのパフォーマンスをテストするためのより長い文です。",
    "素早い茶色のキツネは怠惰な犬を飛び越え、古代の他の寓話にはしばしば道徳的な教訓が含まれています。"
]


def benchmark_gpu(model_id, repeat):
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load model to CPU first for parameter counting and FLOPs analysis if needed before moving to CUDA
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    random.seed(42) # Added for reproducibility
    texts = random.choices(DIVERSE_SENTENCES, k=repeat)
    # Tokenize all texts at once for consistency, will process one by one
    batched_inputs_cpu = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    print(f"Model: {model_id}")
    # Parameter count can be done with model on CPU
    print(f"Parameters: {parameter_count(model)[''] / 1e6:.2f}M")

    # FLOPs analysis with model on CPU and a single input also on CPU
    first_input_cpu_dict = {key: val[0:1] for key, val in batched_inputs_cpu.items()}
    flops = FlopCountAnalysis(model, tuple(first_input_cpu_dict.values()))
    print(f"Estimated FLOPs (single sentence): {flops.total() / 1e9:.2f} GFLOPs")
    
    # Now move model to CUDA for benchmarking
    model.cuda()

    torch.cuda.empty_cache()
    mem_before = get_gpu_memory()

    # Warm-up: process each sentence individually on CUDA
    for i in range(min(3, repeat)): # Warm-up with a few sentences
        single_input_cpu = {key: val[i:i+1] for key, val in batched_inputs_cpu.items()}
        single_input_gpu = {k: v.cuda() for k, v in single_input_cpu.items()}
        with torch.no_grad():
            _ = model(**single_input_gpu)
    torch.cuda.synchronize() # Ensure warm-up is complete

    total_inference_time = 0
    # Timed inference: process each sentence individually on CUDA
    for i in range(repeat):
        single_input_cpu = {key: val[i:i+1] for key, val in batched_inputs_cpu.items()}
        single_input_gpu = {k: v.cuda() for k, v in single_input_cpu.items()}
        
        torch.cuda.synchronize()
        start_time_sentence = time.time()
        with torch.no_grad():
            _ = model(**single_input_gpu)
        torch.cuda.synchronize()
        end_time_sentence = time.time()
        total_inference_time += (end_time_sentence - start_time_sentence)

    mem_after = get_gpu_memory()

    average_inference_time = total_inference_time / repeat if repeat > 0 else 0
    print(f"[GPU] Average Inference Time per sentence: {average_inference_time:.4f} s")
    print(f"[GPU] Total Inference Time for {repeat} sentences: {total_inference_time:.4f} s")
    print(f"[GPU] Memory Used: {mem_after - mem_before:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Benchmark for Hugging Face Model")
    parser.add_argument("--model_id", type=str, default="alexneakameni/language_detection", help="Model ID")
    parser.add_argument("--repeat", type=int, default=50, help="Number of repeated inputs")
    args = parser.parse_args()

    benchmark_gpu(args.model_id, args.repeat)
