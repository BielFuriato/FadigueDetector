# Metricas

## EAR

Eye Aspect Ratio mede a abertura dos olhos usando distancias verticais e horizontais entre landmarks. Valores menores indicam olhos mais fechados. O limiar inicial esta em `config.yaml` como `thresholds.ear_closed`.

## MAR

Mouth Aspect Ratio mede a abertura da boca. Valores maiores indicam boca aberta. Um bocejo so e contado quando o MAR permanece acima do limiar por uma duracao minima.

## PERCLOS

PERCLOS e a proporcao de frames validos com olhos fechados dentro da janela temporal:

```text
PERCLOS = frames com olhos fechados / frames com face detectada
```

Neste prototipo, o PERCLOS e o principal indicador de fadiga ocular.

## Head Pose 2D

A postura da cabeca e estimada por geometria 2D simples:

- inclinacao entre os olhos;
- posicao horizontal do nariz em relacao ao centro da face;
- posicao vertical do nariz entre olhos e queixo.

Essa aproximacao evita `solvePnP` e facilita a execucao local, mas e menos precisa que uma estimativa 3D calibrada.

## Score de Fadiga

Combina PERCLOS, bocejos, fechamento ocular prolongado, duracao media de fechamento e inclinacao da cabeca. O resultado vai de 0 a 100:

- 0 a 30: NORMAL
- 31 a 60: ATENCAO
- 61 a 80: RISCO_MODERADO
- 81 a 100: RISCO_ALTO

## KSS

Karolinska Sleepiness Scale e uma escala subjetiva de sonolencia de 1 a 9. Neste projeto, ela serve apenas para comparacao experimental com os scores calculados.

## PVT

Psychomotor Vigilance Test mede tempo de reacao a estimulos. O script incluso e uma versao terminal simples para apoiar experimentos de atencao/vigilancia.
