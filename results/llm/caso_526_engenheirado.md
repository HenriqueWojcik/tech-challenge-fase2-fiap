## Resumo
O modelo (logistic_regression otimizado por algoritmo genético) classificou o caso como **benigno**, com probabilidade estimada de malignidade de 45.2%.

## Fatores que mais pesaram
- **Simetria (pior valor): 0.3518** - média em casos malignos 0.3235 e em benignos 0.2702; aumenta a chance de malignidade.
- **Suavidade (pior valor): 0.1624** - média em casos malignos 0.1448 e em benignos 0.125; aumenta a chance de malignidade.
- **Raio (erro padrão): 0.1998** - média em casos malignos 0.6091 e em benignos 0.2841; reduz a chance de malignidade.
- **Pontos côncavos (pior valor): 0.1427** - média em casos malignos 0.1822 e em benignos 0.0744; aumenta a chance de malignidade.
- **Compacidade (erro padrão): 0.0144** - média em casos malignos 0.0323 e em benignos 0.0214; aumenta a chance de malignidade.

## Confiança do resultado
Nível de confiança: baixa (zona de incerteza). No conjunto de teste o modelo teve recall de 0.9762 e precisão de 0.9111 para casos malignos.

## Próximos passos sugeridos
- Correlacionar com exame clínico e de imagem.
- Em caso de suspeita ou baixa confiança, considerar biópsia/histopatológico.

## Limitações
Ferramenta de apoio à decisão: não substitui a avaliação do médico. Considera apenas as medidas citológicas do exame, sem histórico clínico, imagem ou fatores de risco.