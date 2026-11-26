# Organizer clarifications

The team reported these answers on Thu 26 Nov 2026. **Dia:** add who answered, and save a screenshot or message link as proof.

| # | Question asked | Answer | Limit we apply | Answered by | Proof |
| --- | --- | --- | --- | --- | --- |
| 1 | Can we train on cloud GPUs (Kaggle, Colab or rented)? | Yes | Team decision: Google Colab only, no Kaggle. Training only. | | |
| 2 | Can the training reward and value network use the engine's true position and game variables? | Yes | Never given to the policy or used in `submission/`. | | |
| 3 | Can the agent estimate relative position from its velocity and turns, using sine and cosine? | Yes | Position estimate and memory map rotation only. No aiming, no `atan2`. | | |
| 4 | Which "common ML utilities" are in the base image? | Yes (the exact list is not recorded) | Check the template's base image Dockerfile before using TensorBoard, matplotlib or tqdm. | | |
| 5 | Can we use AI coding assistants? | Yes | Follow `AGENTS.md`. Be able to explain all code. | | |

An approval covers only the question as written. For anything new, ask again and add a row.
