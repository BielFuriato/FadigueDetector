# Driver Fatigue Distraction Detector

Protótipo acadêmico em Python para detectar sinais visuais de fadiga e distração em motoristas usando webcam, OpenCV, MediaPipe Face Mesh e regras temporais. O sistema calcula EAR, MAR, PERCLOS, score de fadiga, score de distração, emite alertas visuais/sonoros opcionais e salva logs CSV para análise experimental.

> Este projeto não é diagnóstico médico, não substitui avaliação clínica e não deve ser usado como produto de segurança veicular em produção.

## Objetivo Acadêmico

O objetivo é criar uma base funcional e modular para experimentos sobre sonolência e distração. A solução não treina deep learning do zero; ela usa landmarks faciais do MediaPipe e regras temporais inspiradas em métricas comuns de vigilância, como PERCLOS e duração de fechamento ocular.

## Arquitetura

```text
driver-fatigue-distraction-detector/
├── main.py
├── config.yaml
├── src/
│   ├── camera.py
│   ├── face_landmarks.py
│   ├── metrics.py
│   ├── temporal_buffer.py
│   ├── risk_model.py
│   ├── alerts.py
│   ├── logger.py
│   ├── visualization.py
│   └── utils.py
├── experiments/
│   ├── kss_form.py
│   ├── pvt_test.py
│   └── analyze_logs.py
├── data/logs/
└── docs/
```

## Métricas

**EAR** mede a abertura dos olhos. Quando o EAR médio fica abaixo do limiar configurado, o frame é marcado como olho fechado.

**MAR** mede a abertura da boca. Um bocejo é contado quando MAR fica acima do limiar por pelo menos `yawn_min_duration_sec`, com debounce para não contar várias vezes a mesma abertura longa.

**PERCLOS** é a proporção de frames válidos com olhos fechados na janela temporal. O cálculo usa apenas frames em que a face foi detectada.

**Score de Fadiga** combina PERCLOS, bocejos, fechamentos prolongados, duração média de fechamento ocular e inclinação da cabeça.

**Score de Distração** considera tempo olhando para lado, tempo olhando para baixo, ausência de face e rosto fora da posição frontal.

**KSS** é uma escala subjetiva de sonolência de 1 a 9, usada aqui apenas para comparação experimental.

**PVT opcional** é um teste simples de reação para avaliação experimental de atenção/vigilância.

## Instalação

Requisitos: Python 3.10 ou superior.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Linux/macOS, ative o ambiente com:

```bash
source .venv/bin/activate
```

## Como Executar

Com webcam padrão:

```bash
python main.py
```

Com arquivo de vídeo:

```bash
python main.py --source caminho/video.mp4
```

Com outro arquivo de configuração:

```bash
python main.py --config config.yaml
```

Sem som ou sem log:

```bash
python main.py --no-sound
python main.py --no-log
```

Pressione `Q` na janela do OpenCV para encerrar.

## Experimentos

Registrar KSS:

```bash
python experiments/kss_form.py
```

Executar PVT simples:

```bash
python experiments/pvt_test.py
```

Analisar o log mais recente:

```bash
python experiments/analyze_logs.py
```

Analisar um arquivo específico:

```bash
python experiments/analyze_logs.py --file data/logs/session_YYYYMMDD_HHMMSS.csv
```

## Configuração

Todos os limiares principais ficam em `config.yaml`:

```yaml
thresholds:
  ear_closed: 0.20
  mar_open: 0.55
  yawn_min_duration_sec: 1.0
  long_eye_closure_sec: 2.0
  face_missing_sec: 3.0
```

O bloco `scores` controla pesos, referencias e niveis de classificacao sem editar codigo:

```yaml
scores:
  levels:
    attention_min: 31
    moderate_min: 61
    high_min: 81

  fatigue:
    perclos_reference: 0.35
    perclos_weight: 55.0
    yawn_reference_count: 1.0
    yawn_weight: 35.0

  final_decision:
    fatigue_high_min: 61
```

O bloco `validation` controla como os videos classificados viram `alert`, `low_vigilance` ou `drowsy` durante os testes em lote:

```yaml
validation:
  classification:
    warmup_sec: 10
    low_vigilance_min_fatigue: 25
    low_vigilance_min_duration_sec: 2
    drowsy_min_fatigue: 50
    drowsy_min_duration_sec: 2
```

Na validacao, a classificacao simula tempo real com alerta sustentado: apos `warmup_sec`, o score precisa ficar acima do limiar pela duracao minima. Isso evita falsos positivos por um unico frame instavel.

Ajuste os valores conforme camera, usuario, iluminacao, dataset e objetivo experimental.

## Logs

Cada execução cria um CSV em `data/logs/` com colunas como:

- `timestamp`
- `frame_id`
- `face_detected`
- `ear_left`, `ear_right`, `ear_avg`
- `mar`
- `perclos`
- `yawn_count_window`
- `long_eye_closure_count`
- `fatigue_score`
- `distraction_score`
- `final_state`
- `alert_message`
- `fps`

Os gráficos da análise também são salvos em `data/logs/`.

## Limitações

- Sensível a iluminação baixa ou luz muito forte.
- Pode sofrer com óculos, reflexos e oclusões.
- Câmeras ruins ou muito laterais reduzem a qualidade dos landmarks.
- A postura inicial do rosto influencia os indicadores 2D.
- EAR e MAR variam entre pessoas e podem exigir calibração.
- Não é diagnóstico médico.
- É um protótipo acadêmico, não um produto automotivo certificado.

## Próximos Passos

- Criar calibração individual de EAR/MAR nos primeiros segundos.
- Adicionar avaliação com vídeos rotulados.
- Melhorar head pose com `solvePnP`.
- Comparar scores com KSS e PVT em protocolo experimental.
- Adicionar testes unitários para métricas e buffer temporal.
