---
title: Driver Fatigue Detector
sdk: gradio
app_file: app.py
python_version: "3.10"
---

# Driver Fatigue Detector

Protótipo acadêmico em Python para detectar sinais visuais de fadiga em motoristas usando webcam, OpenCV, MediaPipe Face Mesh e regras temporais. O sistema calcula EAR, MAR, PERCLOS, score de fadiga, emite alertas visuais/sonoros opcionais e salva logs CSV para análise experimental.

> Este projeto não é diagnóstico médico, não substitui avaliação clínica e não deve ser usado como produto de segurança veicular em produção.

## Objetivo Acadêmico

O objetivo é criar uma base funcional e modular para experimentos sobre sonolência. A solução não treina deep learning do zero; ela usa landmarks faciais do MediaPipe e regras temporais inspiradas em métricas comuns de vigilância, como PERCLOS e duração de fechamento ocular.

## Arquitetura

```text
driver-fatigue-distraction-detector/
├── main.py
├── config.yaml
├── src/
│   ├── camera.py
│   ├── face_landmarks.py
│   ├── session_processor.py
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

`src/session_processor.py` concentra o processamento por frame. Cada instancia de `FatigueSessionProcessor` possui seu proprio detector, buffer temporal, modelo de risco e gerenciador de alertas, permitindo reuso tanto no fluxo local OpenCV quanto no app web sem compartilhar estado entre sessoes.

## Métricas

**EAR** mede a abertura dos olhos. Quando o EAR médio fica abaixo do limiar configurado, o frame é marcado como olho fechado.

**MAR** mede a abertura da boca. Um bocejo é contado quando MAR fica acima do limiar por pelo menos `yawn_min_duration_sec`, com debounce para não contar várias vezes a mesma abertura longa.

**PERCLOS** é a proporção de frames válidos com olhos fechados na janela temporal. O cálculo usa apenas frames em que a face foi detectada.

**Score de Fadiga** combina PERCLOS, bocejos, fechamentos prolongados, duração média de fechamento ocular e inclinação da cabeça.

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
python3 main.py
```

Com arquivo de vídeo:

```bash
python3 main.py --source caminho/video.mp4
```

Com outro arquivo de configuração:

```bash
python3 main.py --config config.yaml
```

Sem som ou sem log:

```bash
python3 main.py --no-sound
python3 main.py --no-log
```

Pressione `Q` na janela do OpenCV para encerrar.

## App Web com Gradio e FastRTC

O app para Hugging Face Spaces fica em `app.py` e possui duas abas:

- **Video Upload / Samples**: processa um video completo, usa `frame_index / fps` como timestamp e retorna resumo estruturado e video anotado quando a gravacao MP4 estiver disponivel.
- **Live Webcam**: usa FastRTC/WebRTC no navegador, processa frames em tempo real com timestamp de relogio e atualiza metricas durante a transmissao.

Execute localmente:

```bash
python3 app.py
```

Abra o endereco exibido pelo Gradio, normalmente `http://127.0.0.1:7860`.

Os exemplos sao carregados automaticamente de `data/samples/`. Para a versao de demonstracao, mantenha tres videos nessa pasta com extensoes como `.mp4`, `.mov`, `.avi`, `.mkv` ou `.webm`.

Em Hugging Face Spaces, o WebRTC pode precisar de credenciais TURN para funcionar fora de redes locais. O app tenta usar as credenciais suportadas pelo FastRTC quando `HF_TOKEN` ou variaveis Cloudflare TURN estao presentes. Sem TURN, a aba de webcam pode funcionar localmente e falhar em alguns ambientes hospedados.

Com versoes recentes do `mediapipe`, o projeto usa a API `MediaPipe Tasks` e precisa do arquivo `face_landmarker.task`. Na primeira analise, o app tenta baixar automaticamente o modelo oficial para `data/models/face_landmarker.task`. Se o ambiente nao tiver acesso de rede, baixe o modelo manualmente de `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` e salve nesse caminho, ou defina `MEDIAPIPE_FACE_LANDMARKER_MODEL` apontando para o arquivo.

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
- A geração de video anotado no app web depende de codecs disponiveis no ambiente. Se o MP4 nao puder ser criado, a analise textual ainda e retornada.
- A aba de webcam depende de WebRTC/FastRTC e pode exigir configuracao TURN em deploy hospedado.

## Próximos Passos

- Criar calibração individual de EAR/MAR nos primeiros segundos.
- Adicionar avaliação com vídeos rotulados.
- Melhorar head pose com `solvePnP`.
- Comparar scores com KSS e PVT em protocolo experimental.
- Adicionar testes unitários para métricas e buffer temporal.
