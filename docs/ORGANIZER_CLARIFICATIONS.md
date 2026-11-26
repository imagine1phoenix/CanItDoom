# Organizer Clarifications Audit Log

The team recorded these official rulings on Thu 26 Nov 2026. Keep this audit trail updated with official Q&A answers and channels.

| # | Question asked | Answer | Limit we apply | Answered by | Proof / Channel |
| --- | --- | --- | --- | --- | --- |
| 1 | Can we train on cloud GPUs (Kaggle, Colab or rented)? | Yes | Team decision: Google Colab only, no Kaggle. Training only. | DevForge Organizers | Official Discord #announcements / Q&A (Thu 26 Nov) |
| 2 | Can the training reward and value network use the engine's true position and game variables? | Yes | Never given to the policy or used in `submission/`. Privileged state in critic only. | DevForge Organizers | Official Discord #track-1-qa (Thu 26 Nov) |
| 3 | Can the agent estimate relative position from its velocity and turns, using sine and cosine? | Yes | Position estimate and memory map rotation only. No aiming, no `atan2`. | DevForge Organizers | Official Discord #track-1-qa (Thu 26 Nov) |
| 4 | Which "common ML utilities" are in the base image? | Yes (base image verified) | Check template base image Dockerfile before using TensorBoard, matplotlib or tqdm. Training only. | DevForge Organizers | Official Starter Template Dockerfile & Lockfile |
| 5 | Can we use AI coding assistants? | Yes | Follow `AGENTS.md`. Maintain team member ownership and be able to defend all code in judge defense. | DevForge Organizers | Official DevForge 2026 Participant Guidelines |

An approval covers only the question as written. For any new question or organizer ruling, ask Prit and record a new row immediately.
