# doom-lx — Doom dentro dos pesos (linuxformer LX)

O clipe de Doom jogado e renderizado pelo circuito do Qwenputer, na máquina LX
do linuxformer (RV32IM, ~1,5 tokens por instrução). Código privado.

| pasta | o que é |
|---|---|
| `port/` | o port do DoomGeneric para a LX (submodule, histórico próprio): `port-lf/build.sh`, emulador nativo `native/fastemu-lf.c`, patch das otimizações `port-lf/doomgeneric-lf.patch` |
| `ops/work/` | operação: `STATE.md` (estado completo e honesto), `runs/` (drivers de GPU + notebook), `kaggle/` (runner e empacotamento), `modal_*.py` (runs na Modal) |
| `CREDITS.md` | licenças e atribuições de tudo que é de terceiros |

## Estado atual

**Dois clipes entregues, todo frame verificado byte a byte contra `fastemu-lf`:**

| clipe | frames | taxa | verificação | custo |
|---|---|---|---|---|
| 96x60, L40S | 116 (6..121) | 17.700 tok/s | 116/116 exatos | US$ 3,87 |
| 192x120, A100 40GB | 116 (6..121) | 19.109 tok/s | 116/116 exatos | ~10 CU |

O custo por frame cresce devagar com a resolução — 192x120 tem 4x os pixels de 96x60
mas custa só 2,03x por frame, porque o engine domina e o desenho de pixel é minoria.
Detalhes e a tabela completa: `ops/work/STATE.md` seções 13 a 16, com os relatórios de
cada run em `ops/work/RUNLOG-l40s.md` e `ops/work/RUNLOG-a100-192x120.md`.

Os clipes e as folhas de contato estão em `ops/work/preview/`:

* `clip-e1m5-in-weights-192x120.mp4` — E1M5, frames 6..121, 11,6 s
* `clip-e1m5-in-weights-96x60.mp4` — o mesmo trecho na build menor
* `contact-sheet-192x120.png`, `contact-sheet-96x60.png`

## O que não é verdade (e já foi corrigido no ledger)

* **Não é 0 erro.** ~4,2e-6 das previsões erram (394 + 28 + 28 em 106,1M tokens). Não
  afeta os pixels, que são exatos, mas retira a afirmação "prevê todo token" das
  sessões T4. O padrão é par de um a menos (`65536`/`65537`).
* **A A100 não é mais lenta que a L40S.** Uma leitura intermediária dizia isso; estava
  medida com artefatos trocados. Com o par certo, 19.109 contra 17.700 tok/s.
* **Os clipes não são 320x200.** São 96x60 e 192x120, com `screenblocks=11`.

## Orçamento por tokens, não por constantes

| build | instr/frame | tokens/frame | tokens até o frame 121 |
|---|---|---|---|
| 96x60 | 226.105 | 341 mil | ~48M |
| 192x120 | 459.923 | 694 mil | **~106M** |

O `tpi` real do build de 192x120 é **1,606**, não a constante 1,5087 usada antes — foi
por isso que um orçamento de 95M parou no frame 119.

## Como rodar

`ops/work/runs/README.md` tem os comandos, e `ops/work/runs/run-192x120.ipynb` a
mesma coisa em notebook. O resumo:

```bash
colab new -s doom192 -gpu A100
colab exec -s doom192 -f runs/colab_setup.py --timeout 1700
colab exec -s doom192 -f runs/colab_run.py --timeout 9000 \
    --env HF_TOKEN=... --env HF_DATASET=<owner>/<dataset> \
    --env SET=192 --env RESUME=0 --env HOURS=3.0 --env NTOK=95000000
```

O `packed.pt` é por build: `.text` diferente é conjunto de pesos diferente, e um
checkpoint só retoma nos pesos com que foi salvo.
