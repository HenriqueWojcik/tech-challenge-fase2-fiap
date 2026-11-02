## Resumo
O modelo (logistic_regression otimizado por algoritmo genético) classificou o caso como **benigno**, com probabilidade estimada de malignidade de 16.9%.

## Fatores que mais pesaram
- **Textura (pior valor): 20.86** - média em casos malignos 29.3182 e em benignos 23.5151; reduz a chance de malignidade.
- **Textura (média): 15.79** - média em casos malignos 21.6049 e em benignos 17.9148; reduz a chance de malignidade.
- **Pontos côncavos (pior valor): 0.1383** - média em casos malignos 0.1822 e em benignos 0.0744; aumenta a chance de malignidade.
- **Simetria (pior valor): 0.2589** - média em casos malignos 0.3235 e em benignos 0.2702; reduz a chance de malignidade.
- **Raio (erro padrão): 0.2787** - média em casos malignos 0.6091 e em benignos 0.2841; reduz a chance de malignidade.

## Confiança do resultado
Nível de confiança: moderada. No conjunto de teste o modelo teve recall de 0.9762 e precisão de 0.9111 para casos malignos.

## Próximos passos sugeridos
- Correlacionar com exame clínico e de imagem.
- Em caso de suspeita ou baixa confiança, considerar biópsia/histopatológico.

## Limitações
Ferramenta de apoio à decisão: não substitui a avaliação do médico. Considera apenas as medidas citológicas do exame, sem histórico clínico, imagem ou fatores de risco.