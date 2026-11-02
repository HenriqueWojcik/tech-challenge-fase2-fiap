## Resumo
O modelo (logistic_regression otimizado por algoritmo genético) classificou o caso como **benigno**, com probabilidade estimada de malignidade de 0.0%.

## Fatores que mais pesaram
- **Textura (pior valor): 14.82** - média em casos malignos 29.3182 e em benignos 23.5151; reduz a chance de malignidade.
- **Suavidade (pior valor): 0.0931** - média em casos malignos 0.1448 e em benignos 0.125; reduz a chance de malignidade.
- **Pontos côncavos (pior valor): 0.0319** - média em casos malignos 0.1822 e em benignos 0.0744; reduz a chance de malignidade.
- **Textura (média): 12.35** - média em casos malignos 21.6049 e em benignos 17.9148; reduz a chance de malignidade.
- **Simetria (pior valor): 0.2143** - média em casos malignos 0.3235 e em benignos 0.2702; reduz a chance de malignidade.

## Confiança do resultado
Nível de confiança: alta. No conjunto de teste o modelo teve recall de 0.9762 e precisão de 0.9111 para casos malignos.

## Próximos passos sugeridos
- Correlacionar com exame clínico e de imagem.
- Em caso de suspeita ou baixa confiança, considerar biópsia/histopatológico.

## Limitações
Ferramenta de apoio à decisão: não substitui a avaliação do médico. Considera apenas as medidas citológicas do exame, sem histórico clínico, imagem ou fatores de risco.