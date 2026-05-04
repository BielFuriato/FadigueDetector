# Metodologia

Este projeto implementa um prototipo academico para observar sinais visuais associados a fadiga e distracao em motoristas. A abordagem usa webcam, OpenCV, MediaPipe Face Mesh e regras temporais. Nenhuma rede neural e treinada do zero.

## Fluxo Geral

1. A camera ou o video e aberto pelo modulo `CameraStream`.
2. Cada frame e processado pelo MediaPipe Face Mesh.
3. Os landmarks faciais sao convertidos para coordenadas de pixel.
4. O sistema calcula EAR, MAR e indicadores 2D de postura da cabeca.
5. O `TemporalBuffer` acumula os ultimos segundos de observacao.
6. O modelo de risco combina PERCLOS, bocejos, fechamentos prolongados, postura e ausencia de face.
7. A interface OpenCV exibe os indicadores e os alertas.
8. O logger salva um CSV por execucao para analise posterior.

## Decisoes de Projeto

- O MediaPipe Face Mesh foi escolhido por ser estavel, local e simples para webcam.
- O PERCLOS e calculado apenas com frames em que a face foi detectada.
- Bocejos usam MAR acima do limiar por duracao minima, com debounce para evitar multiplas contagens na mesma abertura de boca.
- Piscadas curtas e fechamentos prolongados sao separados pela duracao do olho fechado.
- Distração usa olhar lateral, olhar para baixo, rosto nao frontal e ausencia de face.

## Limitacoes

Os indicadores sao aproximacoes visuais. Eles podem variar com iluminacao, oculos, camera, distancia, angulo do rosto e diferencas individuais. O sistema nao e diagnostico medico nem ferramenta clinica.
