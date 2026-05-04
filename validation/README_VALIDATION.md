# Validacao com Videos Locais

Este modulo permite validar o pipeline atual do projeto com videos classificados manualmente em pastas. Ele nao baixa datasets, nao usa Kaggle API, nao treina redes neurais e nao altera o modelo principal. A ideia e medir como as regras atuais baseadas em OpenCV, MediaPipe, EAR, MAR, PERCLOS e scores temporais se comportam em videos de fadiga, baixa vigilancia, bocejo, microssono e distracao.

## Estrutura Esperada

Baixe os datasets manualmente e coloque os videos nas pastas abaixo:

```text
data/test_videos/
├── uta_rldd/
│   ├── alert/
│   ├── low_vigilance/
│   └── drowsy/
├── nitymed/
│   ├── yawning/
│   └── microsleep/
├── yawdd/
│   ├── normal/
│   ├── talking/
│   └── yawning/
└── own_recordings/
    ├── normal/
    ├── eyes_closed/
    ├── yawn/
    ├── looking_down/
    ├── looking_side/
    └── no_face/
```

Pastas vazias sao permitidas. O script simplesmente ignora classes sem videos.

## Gerar Manifest

O manifest e um CSV com um video por linha e metadados esperados:

```bash
python validation/build_manifest.py --videos-dir data/test_videos --output data/validation/manifests/validation_manifest.csv
```

Colunas geradas:

```text
video_id, video_path, dataset, subject_id, expected_class, expected_event,
event_start_sec, event_end_sec, notes
```

Como os datasets podem estar classificados por video inteiro, `event_start_sec` e `event_end_sec` ficam vazios inicialmente.

## Rodar Validacao

```bash
python validation/batch_validate.py --manifest data/validation/manifests/validation_manifest.csv --config config.yaml --no-display
```

Parametros uteis:

```bash
python validation/batch_validate.py --manifest data/validation/manifests/validation_manifest.csv --no-display --max-videos 5
python validation/batch_validate.py --manifest data/validation/manifests/validation_manifest.csv --no-display --dataset-filter UTA-RLDD
python validation/batch_validate.py --manifest data/validation/manifests/validation_manifest.csv --no-display --class-filter drowsy,yawning
```

Sem `--no-display`, uma janela OpenCV mostra o processamento.

## Saidas

Para cada video:

```text
data/validation/outputs/{video_id}_frames.csv
```

Resumo consolidado:

```text
data/validation/outputs/validation_summary.csv
```

Relatorio Markdown:

```text
data/validation/reports/validation_report.md
```

Graficos:

```text
data/validation/plots/fatigue_score_by_class.png
data/validation/plots/perclos_by_class.png
data/validation/plots/confusion_matrix.png
data/validation/plots/timeline_{video_id}.png
```

Se um video falhar, o erro e registrado em `processing_error` e o lote continua.

## Como Interpretar

**Sonolencia detectada** significa que o score de fadiga chegou a risco moderado ou alto, normalmente por PERCLOS alto, fechamento prolongado, bocejo ou combinacao desses sinais.

**Baixa vigilancia** representa uma zona preventiva. O video nao precisa chegar a risco alto; basta atingir ATENCAO para indicar que o sistema percebeu deterioracao inicial.

**Deteccao preventiva** e marcada como verdadeira quando videos `low_vigilance` atingem ATENCAO, ou quando videos `drowsy`/`microsleep` atingem risco moderado ou alto.

**Falso positivo** ocorre quando videos esperados como alertas/normais geram alerta forte, especialmente fadiga moderada/alta sem necessidade.

**Falso negativo** ocorre quando um evento esperado, como bocejo, microssono ou sonolencia, nao aparece nos indicadores do sistema.

## Calibracao pelo config.yaml

Os gatilhos da validacao ficam no bloco `validation` de `config.yaml`. Por exemplo:

```yaml
validation:
  classification:
    warmup_sec: 10
    low_vigilance_min_fatigue: 25
    low_vigilance_min_duration_sec: 2
    drowsy_min_fatigue: 50
    drowsy_min_duration_sec: 2
```

A classificacao do video usa alerta sustentado, simulando tempo real: depois de `warmup_sec`, o score precisa cruzar o limiar por pelo menos a duracao minima. A media continua no relatorio apenas como contexto.

Se quiser que um video vire `drowsy` com score menor, reduza `drowsy_min_fatigue`. Para mudar o calculo do score em si, ajuste o bloco `scores.fatigue`, especialmente `perclos_reference`, `perclos_weight`, `yawn_weight` e `long_closure_weight`.

## Observacoes

- Use os resultados para validacao e calibracao, nao para treinar um novo modelo.
- Limiar de EAR, MAR, PERCLOS e duracoes podem precisar de ajuste por camera, iluminacao e sujeito.
- Videos de datasets diferentes podem ter enquadramento e qualidade muito distintos.
- A matriz de confusao e aproximada porque algumas classes sao eventos especificos, enquanto o detector atual gera scores de risco.
