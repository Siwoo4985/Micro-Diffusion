# Micro-Diffusion
A minimal discrete text diffusion model in pure Python. Like Karpathy's MicroGPT showed GPT's essence in ~200 lines, Micro Diffusion shows how text diffusion works — generating all tokens at once by iteratively unmasking from noise, instead of left-to-right. Trains on 32K names, runs on CPU in minutes. No framework required.
