conda activate LocalLLM

llama-server `
  -m "C:\Users\locng\OneDrive\Máy tính\LLMModels\microsoft_Phi-4-mini-instruct-Q6_K_L.gguf" `
  -c 8192 `
  -ngl 999 `
  --host 0.0.0.0 `
  --port 8080