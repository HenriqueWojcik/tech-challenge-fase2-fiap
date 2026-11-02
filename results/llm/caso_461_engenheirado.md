## Resumo
O modelo (logistic_regression otimizado por algoritmo genético) classificou o caso como **maligno**, com probabilidade estimada de malignidade de 100.0%.

## Fatores que mais pesaram
- **Área (erro padrão): 542.2** - média em casos malignos 72.6724 e em benignos 21.1351; aumenta a chance de malignidade.
- **Raio (erro padrão): 2.547** - média em casos malignos 0.6091 e em benignos 0.2841; aumenta a chance de malignidade.
- **Área (pior valor): 4254.0** - média em casos malignos 1422.2863 e em benignos 558.8994; aumenta a chance de malignidade.
- **Perímetro (erro padrão): 18.65** - média em casos malignos 4.3239 e em benignos 2.0003; aumenta a chance de malignidade.
- **Raio (pior valor): 36.04** - média em casos malignos 21.1348 e em benignos 13.3798; aumenta a chance de malignidade.

## Confiança do resultado
Nível de confiança: alta. No conjunto de teste o modelo teve recall de 0.9762 e precisão de 0.9111 para casos malignos.

## Próximos passos sugeridos
- Correlacionar com exame clínico e de imagem.
- Em caso de suspeita ou baixa confiança, considerar biópsia/histopatológico.

## Limitações
Ferramenta de apoio à decisão: não substitui a avaliação do médico. Considera apenas as medidas citológicas do exame, sem histórico clínico, imagem ou fatores de risco.