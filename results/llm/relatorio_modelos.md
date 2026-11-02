## Relatório (modo offline)
Configure uma LLM no .env para gerar este relatório.

| modelo              | abordagem              |   cv_f2 |   teste_recall_maligno |   teste_precision_maligno |   teste_falsos_negativos |   treinos |
|:--------------------|:-----------------------|--------:|-----------------------:|--------------------------:|-------------------------:|----------:|
| random_forest       | original_fase1         |  0.9517 |                 0.9286 |                    0.9512 |                        3 |         1 |
| random_forest       | E1_base                |  0.9627 |                 0.9286 |                    0.907  |                        3 |        92 |
| random_forest       | E2_mutacao_alta        |  0.9628 |                 0.9524 |                    0.9091 |                        2 |       138 |
| random_forest       | E3_populacao_maior     |  0.965  |                 0.9286 |                    0.8864 |                        3 |       160 |
| random_forest       | E4_roleta_sem_elitismo |  0.9603 |                 0.9286 |                    0.907  |                        3 |       130 |
| random_forest       | E5_semeado_fase1       |  0.9697 |                 0.9286 |                    0.8864 |                        3 |       132 |
| random_forest       | busca_aleatoria        |  0.9592 |                 0.9524 |                    0.8696 |                        2 |        92 |
| logistic_regression | original_fase1         |  0.9668 |                 0.9762 |                    0.9762 |                        1 |         1 |
| logistic_regression | E1_base                |  0.9753 |                 0.9762 |                    0.9111 |                        1 |        53 |
| logistic_regression | E2_mutacao_alta        |  0.9753 |                 0.9762 |                    0.9111 |                        1 |        87 |
| logistic_regression | E3_populacao_maior     |  0.9753 |                 0.9762 |                    0.9111 |                        1 |        92 |
| logistic_regression | E4_roleta_sem_elitismo |  0.9704 |                 0.9762 |                    0.9762 |                        1 |        79 |
| logistic_regression | E5_semeado_fase1       |  0.9742 |                 0.9762 |                    0.9111 |                        1 |        52 |
| logistic_regression | busca_aleatoria        |  0.9753 |                 0.9762 |                    0.9111 |                        1 |        53 |