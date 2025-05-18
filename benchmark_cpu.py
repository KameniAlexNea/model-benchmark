import time
import psutil
import argparse
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count

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


def benchmark_cpu(model_id, repeat):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    random.seed(42)  # Added for reproducibility
    texts = random.choices(DIVERSE_SENTENCES, k=repeat)
    # Tokenize all texts at once for consistency in padding, but process one by one
    batched_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    print(f"Model: {model_id}")
    print(f"Parameters: {parameter_count(model)[''] / 1e6:.2f}M")

    # Prepare a single input for FLOPs analysis (e.g., the first sentence)
    # FlopCountAnalysis expects a tuple of inputs as passed to model.forward()
    # So, we create a dictionary for a single input similar to how model expects it.
    first_input_dict = {key: val[0:1] for key, val in batched_inputs.items()}
    flops = FlopCountAnalysis(model, tuple(first_input_dict.values()))
    print(f"Estimated FLOPs (single sentence): {flops.total() / 1e9:.2f} GFLOPs")

    mem_before = psutil.Process().memory_info().rss / (1024 ** 2)

    # Warm-up: process each sentence individually
    for i in range(min(3, repeat)): # Warm-up with a few sentences
        single_input = {key: val[i:i+1] for key, val in batched_inputs.items()}
        with torch.no_grad():
            _ = model(**single_input)

    total_inference_time = 0
    # Timed inference: process each sentence individually
    for i in range(repeat):
        single_input = {key: val[i:i+1] for key, val in batched_inputs.items()}
        start_time_sentence = time.time()
        with torch.no_grad():
            _ = model(**single_input)
        end_time_sentence = time.time()
        total_inference_time += (end_time_sentence - start_time_sentence)

    mem_after = psutil.Process().memory_info().rss / (1024 ** 2)

    average_inference_time = total_inference_time / repeat if repeat > 0 else 0
    print(f"[CPU] Average Inference Time per sentence: {average_inference_time:.4f} s")
    print(f"[CPU] Total Inference Time for {repeat} sentences: {total_inference_time:.4f} s")
    print(f"[CPU] Memory Used: {mem_after - mem_before:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Benchmark for Hugging Face Model")
    parser.add_argument("--model_id", type=str, default="alexneakameni/language_detection", help="Model ID")
    parser.add_argument("--repeat", type=int, default=50, help="Number of repeated inputs")
    args = parser.parse_args()

    benchmark_cpu(args.model_id, args.repeat)
