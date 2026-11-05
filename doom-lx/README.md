# doom-lx — Doom dentro dos pesos (linuxformer LX)

O clipe de Doom jogado e renderizado pelo circuito do Qwenputer, na máquina LX
do linuxformer (RV32IMA, 1,5 tokens por instrução). Código privado.

| pasta | o que é |
|---|---|
| `port/` | o port do DoomGeneric para a LX (repo git próprio, histórico intacto): `port-lf/build.sh`, emulador nativo `native/fastemu-lf.c`, patch das otimizações `port-lf/doomgeneric-lf.patch` |
| `ops/work/` | operação: `STATE.md` (estado completo e honesto), `run_clip.sh` (um comando, resumível, com teto de orçamento), `modal_doom.py` (run na Modal), `clip_from_weights.py` (monta o vídeo) |

Estado atual (detalhes em `ops/work/STATE.md`): boot 4,56M instruções, 344.945
tokens/frame de gameplay (skip=1) ou ~416k por frame cobrindo 3 tics (skip=3,
tempo quase real). Estado já pago no Volume `doom-lx-packed`: token 3.403.776 do
boot, 0 erros. O gargalo de GPU (cabeça relendo `w_head` a cada 4096 tokens) está
consertado no `lx/lx/fast.py` do repo linuxformer, mas o ganho de 8x ainda não foi
medido.

Próximo passo quando houver compute: `ops/work/run_clip.sh 0.45` mede a
velocidade nova; com ela confirmada, `run_clip.sh <orçamento> 50` produz os
frames do clipe.
