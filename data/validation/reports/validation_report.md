# Relatorio de Validacao

## Resumo

- Total de videos no manifest: 18
- Total de videos processados: 18
- Videos com erro: 0
- Acuracia basica: 83.3%
- Falsos positivos: 0
- Falsos negativos: 3
- Taxa de deteccao preventiva para low_vigilance: 83.3%

## Quantidade por Dataset

| dataset | count |
| --- | --- |
| UTA-RLDD | 18 |

## Quantidade por Classe Esperada

| expected_class | count |
| --- | --- |
| alert | 6 |
| drowsy | 6 |
| low_vigilance | 6 |

## Quantidade por Classe Prevista

| predicted_class | count |
| --- | --- |
| alert | 8 |
| drowsy | 7 |
| low_vigilance | 3 |

## PERCLOS por Classe

| expected_class | mean | max |
| --- | --- | --- |
| alert | 0.1161433964702656 | 0.1818181818181818 |
| drowsy | 0.5335520220658331 | 1.0 |
| low_vigilance | 0.5470478060141387 | 1.0 |

## Fatigue Score por Classe

| expected_class | mean | max |
| --- | --- | --- |
| alert | 27.208162959822058 | 43.00769817711922 |
| drowsy | 64.14311387325837 | 84.06684529269283 |
| low_vigilance | 61.5366952335092 | 83.74954597720121 |

## Melhores Videos

| video_id | dataset | expected_class | predicted_class | max_fatigue_score | first_drowsiness_time_sec | is_correct_basic |
| --- | --- | --- | --- | --- | --- | --- |
| uta_rldd_drowsy_10_5 | UTA-RLDD | drowsy | drowsy | 84.06684529269283 | 418.9561972299169 | True |
| uta_rldd_low_vigilance_5_3 | UTA-RLDD | low_vigilance | drowsy | 83.74954597720121 | 189.47161067300425 | True |
| uta_rldd_drowsy_10_3 | UTA-RLDD | drowsy | drowsy | 83.30073460163041 | 11.001785714285711 | True |
| uta_rldd_low_vigilance_5_6 | UTA-RLDD | low_vigilance | drowsy | 82.95548088080504 | 350.58217016086 | True |
| uta_rldd_drowsy_10 | UTA-RLDD | drowsy | drowsy | 76.46255367645324 | 119.42386374421554 | True |
| uta_rldd_drowsy_10_6 | UTA-RLDD | drowsy | drowsy | 68.72892580951344 | 314.02689876977536 | True |
| uta_rldd_low_vigilance_5_2 | UTA-RLDD | low_vigilance | drowsy | 62.94237241981043 | 2.002077003811523 | True |
| uta_rldd_low_vigilance_5_4 | UTA-RLDD | low_vigilance | low_vigilance | 53.91324763677558 |  | True |

## Piores Videos

| video_id | dataset | expected_class | expected_event | predicted_class | first_low_vigilance_time_sec | first_drowsiness_time_sec | false_positive_flag | false_negative_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uta_rldd_drowsy_10_2 | UTA-RLDD | drowsy | drowsiness | low_vigilance | 3.503630545133525 |  | False | True |
| uta_rldd_low_vigilance_5 | UTA-RLDD | low_vigilance | early_drowsiness | alert |  |  | False | True |
| uta_rldd_drowsy_10_4 | UTA-RLDD | drowsy | drowsiness | alert |  |  | False | True |

## Videos com Erro

Nenhum item.


## Graficos

- `fatigue_score_by_class.png`
- `perclos_by_class.png`
- `confusion_matrix.png`
- `timeline_{video_id}.png` para cada video processado

Matriz de confusao salva em `data\validation\plots\confusion_matrix.png`.

## Observacoes Automaticas

- Ha falsos negativos; alguns eventos esperados nao atingiram os limiares atuais.
- A deteccao preventiva em baixa vigilancia foi consistente nesta amostra.
