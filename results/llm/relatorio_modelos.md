## Principais conclusões  
- **Maior recall e menor taxa de falsos negativos**: a regressão logística (todas as abordagens) alcançou recall de 0,9762, enquanto a floresta aleatória nunca ultrapassou 0,9524.  
- **Precisão mais alta em regressão logística**: 0,9762 em todas as abordagens, comparado a 0,8864‑0,9512 na floresta aleatória.  
- **Falsos negativos**: regressão logística apresentou apenas 1 falso negativo em 114 pacientes, enquanto a floresta aleatória teve de 2 a 3.  
- **Robustez do modelo**: a regressão logística manteve desempenho consistente em todas as estratégias de otimização, com variação mínima no F2‑score (0,9668‑0,9753).  
- **Custo computacional**: a regressão logística requer muito menos iterações de treinamento (1‑79) em comparação com a floresta aleatória (1‑160), reduzindo tempo e recursos.

## Qual modelo recomendamos e por quê  
Recomendamos a **regressão logística** como modelo de produção. Ela oferece:

1. **Maior sensibilidade** (recall 97,6 %) → menor risco de deixar cânceres passarem despercebidos.  
2. **Precisão equivalente ou superior** (até 97,6 %) → menos alarmes falsos que poderiam sobrecarregar a equipe clínica.  
3. **Estabilidade**: desempenho quase idêntico em todas as abordagens de otimização, indicando menor risco de overfitting.  
4. **Eficiência**: menos recursos computacionais e menor tempo de treinamento, facilitando atualizações e manutenção.

## O que significa para os pacientes  
- **Redução de falsos negativos**: com apenas 1 falso negativo em 114 exames, a probabilidade de um câncer ser não detectado cai para < 1 %.  
- **Maior confiança no diagnóstico**: pacientes e médicos podem confiar que a maioria dos casos malignos será identificada, reduzindo a necessidade de exames complementares desnecessários.  
- **Menos ansiedade**: menos alarmes falsos (alta precisão) evita consultas de seguimento desnecessárias e o estresse associado.

## Cuidados antes de usar em produção  
1. **Validação externa**: testar o modelo em bases de dados de outros hospitais ou regiões para garantir generalização.  
2. **Monitoramento contínuo**: acompanhar métricas de recall, precisão e taxa de falsos negativos em tempo real; implementar alertas de drift de dados.  
3. **Calibração de threshold**: ajustar o ponto de corte se a política clínica mudar (ex.: priorizar ainda mais a sensibilidade).  
4. **Interpretação clínica**: disponibilizar explicações (coeficientes, SHAP) para que os radiologistas entendam as