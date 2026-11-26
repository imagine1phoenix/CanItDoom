# Map sources

List every training or held-out map that our own generator did not create.

| WAD or map | Source link | Author | Licence | Used for (train / held-out) |
| --- | --- | --- | --- | --- |
| `maps/two_rooms.wad` (`MAP01`) | Official starter template (`devforge-doom-template`) | DevForge organizers | Template licence | `train` (Stage 1 smoke test) |
| `freedoom2.wad` (`MAP02`, `MAP06`, `MAP09`, `MAP14`, `MAP20`, `MAP27`) | https://freedoom.github.io/ (bundled with ViZDoom 1.3.3) | Freedoom project contributors | BSD-3-Clause | `held_out` (6 maps, strictly excluded from training) |
| `freedoom2.wad` (`MAP01`, `MAP03`–`MAP05`, `MAP07`–`MAP08`, `MAP10`–`MAP13`, `MAP15`–`MAP19`, `MAP21`–`MAP26`, `MAP28`–`MAP32`) | https://freedoom.github.io/ (bundled with ViZDoom 1.3.3) | Freedoom project contributors | BSD-3-Clause | `train` (Stage 5 generalization, 26 maps) |
