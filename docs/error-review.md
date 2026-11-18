# Manual review of 30 base-model errors (CROHME 2019)

The 30 cases are the seeded sample from `wlm.compare --review 30`
(`results/error_analysis/base_2019/review.html`). Each case was checked by
looking at the image, the true label and the base prediction.

## Tags

- **Glyph:** the writer's shape for a symbol looks like another symbol, and the
  model picks the other one. This is what personalization should fix.
- **Structure:** the writing is clear, but the model gets the layout wrong
  (root index, nesting, subscripts) or loses count of repeated symbols.
- **Label/image:** the image does not clearly match the label, or the
  prediction matches what is actually written.

## Summary

| Tag | Cases | Share |
|---|---|---|
| Glyph | 19 | 63% |
| Structure | 7 | 23% |
| Label/image | 4 | 13% |

Most errors come from how one person writes a symbol, not from hard math
structure. That supports the project's idea: a few samples from a writer
should teach the model that writer's shapes.

## Cases

| # | Image id | Error | Tag | Note |
|---|---|---|---|---|
| 1 | UN19wb_1119_em_1164 | `r` → `y` | Glyph | r drawn with a tail |
| 2 | UN19_1029_em_415 | `r`→`x`, `E`→`t`, `l`→`p`, `V`→`v` | Glyph | r like π, E like ∈, l like P |
| 3 | UN19wb_1113_em_1065 | repeated `d x` terms reordered | Structure | long run of similar terms |
| 4 | UN19_1032_em_454 | `l` → `f` | Glyph | cursive l |
| 5 | ISICal19_1205_em_817 | `C` → `c` | Glyph | capital written small |
| 6 | UN19_1017_em_238 | root index `n` lost | Structure | tiny index |
| 7 | UN19_1039_em_561 | `r`→`\gamma`, `\ldots`→`\cdots` | Glyph | r like γ, dots raised |
| 8 | UN19_1037_em_530 | `z` → subscript `2` | Glyph | small z sits low |
| 9 | UN19_1031_em_440 | `t` → `+` | Glyph | t like a plus sign |
| 10 | UN19wb_1114_em_1089 | `\theta`→`0`, `\phi`→`\theta` | Glyph | θ like 0, φ like ∅ |
| 11 | UN19wb_1118_em_1150 | commas → `1` / `+` | Glyph | commas drawn as short strokes |
| 12 | UN19_1021_em_287 | `4` → `( 1` | Glyph | open 4 |
| 13 | UN19_1036_em_520 | `\times` → `x` | Glyph | small × |
| 14 | UN19_1026_em_366 | `,` → `-` | Glyph | flat comma |
| 15 | UN19_1046_em_667 | `A`→`t`, `X`→`x`, `8`→`1` | Glyph | A like t |
| 16 | UN19_1015_em_195 | `z`→`g`, `P`→`p` | Glyph | z with descender |
| 17 | UN19_1039_em_555 | brackets moved, `5`→`3` | Structure | nested roots |
| 18 | UN19_1004_em_54 | superscript `-p` read inline | Structure | script barely raised |
| 19 | UN19_1019_em_263 | `a` → `d` | Glyph | tall a |
| 20 | UN19wb_1112_em_1058 | `\rightarrow` → `>` | Glyph | short arrow |
| 21 | ISICal19_1211_em_892 | `( 4 \pi ) ^ { 4 }` broken | Structure | power on a bracket |
| 22 | UN19_1049_em_719 | one `0` of five lost | Structure | counting repeats |
| 23 | UN19wb_1116_em_1113 | `(` → `+` | Glyph | small bracket |
| 24 | UN19_1017_em_229 | `f _ { n }` → `8 o` | Label/image | image is unreadable |
| 25 | UN19_1041_em_594 | `X`→`x`, superscript lost | Glyph | case confusion |
| 26 | UN19wb_1108_em_1003 | `j` → `i` | Glyph | short j |
| 27 | UN19wb_1116_em_1124 | `t ^ { 2 }` inside the root | Label/image | image shows it inside |
| 28 | UN19_1047_em_687 | extra `,` | Label/image | image has a comma |
| 29 | UN19_1006_em_76 | subscript `m n` lost | Structure | tiny subscript |
| 30 | UN19_1021_em_298 | English words | Label/image | text, not math |
