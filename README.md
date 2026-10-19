# Precificação Dinâmica com RLlib

<p align="justify">Este projeto desenvolve uma simulação de precificação dinâmica para um serviço de transporte urbano. Um agente aprende a escolher entre diferentes multiplicadores de tarifa com base na procura estimada, na quantidade de motoristas disponíveis e no horário do dia. O objetivo definido no código é aumentar a receita das corridas realizadas e reduzir as perdas associadas à espera de passageiros.</p>

## 1. Objetivo do projeto

<p align="justify">Em vez de manter uma tarifa fixa durante todo o dia, o sistema permite que o preço varie conforme as condições simuladas. Em horários de maior procura, por exemplo, pode ser vantajoso aplicar uma tarifa maior. Entretanto, preços excessivos reduzem a procura atendida, enquanto preços baixos em períodos de procura elevada podem deixar de aproveitar oportunidades de receita. O agente aprende a equilibrar esses fatores por meio de tentativas realizadas no ambiente simulado.</p>

<p align="justify">O projeto utiliza a biblioteca RLlib, da plataforma Ray, para treinar um agente com o método de otimização de política proximal, conhecido pela sigla PPO. O ambiente é construído com a biblioteca Gymnasium, enquanto NumPy realiza os cálculos numéricos e PyTorch é utilizado para executar a política treinada durante o teste.</p>

## 2. Construção do ambiente de simulação

<p align="justify">A classe <code>AmbienteTarifaDinamica</code> representa o cenário em que o agente toma decisões. Ela define as informações que o agente recebe, as tarifas que pode escolher, a forma como o ambiente muda ao longo do dia e a recompensa recebida após cada decisão.</p>

```python
class AmbienteTarifaDinamica(gym.Env):
    def __init__(self, config={}):
        super().__init__()
        self.max_motoristas = 50
        self.max_demanda = 50

        self.precos_multiplicadores = [
            0.8, 1.0, 1.2, 1.4, 1.6,
            1.8, 2.0, 2.2, 2.5
        ]

        self.action_space = spaces.Discrete(
            len(self.precos_multiplicadores)
        )

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )
```

<p align="justify"><b>Explicação:</b> o agente pode escolher entre nove multiplicadores, que vão de 0,8 a 2,5 vezes a tarifa de referência. A observação contém três valores: procura, motoristas disponíveis e horário. Esses valores são normalizados para uma faixa próxima de zero a um, facilitando sua utilização pelo modelo.</p>

## 3. Representação do estado

<p align="justify">O estado resume as condições do serviço em determinado momento. A função <code>_obter_observacao()</code> divide a procura máxima, a quantidade máxima de motoristas e as 24 horas do dia para produzir valores normalizados. Assim, o agente não recebe apenas números em escalas muito diferentes.</p>

```python
def _obter_observacao(self):
    return np.array([
        self.demanda / self.max_demanda,
        self.motoristas / self.max_motoristas,
        self.hora / 24.0
    ], dtype=np.float32)
```

<p align="justify">A procura e a disponibilidade de motoristas são simuladas conforme o horário. O código estabelece faixas de procura mais elevadas pela manhã, entre 7h e 9h, e no fim da tarde, entre 17h e 19h. Durante a madrugada, entre 1h e 5h, os valores simulados são menores. Nos demais períodos, são utilizados valores intermediários. Como os números são sorteados, a mesma hora pode apresentar condições diferentes em execuções distintas.</p>

## 4. Escolha da tarifa e cálculo da recompensa

<p align="justify">A função <code>step()</code> é a parte central do ambiente. Ela recebe a ação escolhida pelo agente, converte o multiplicador em preço, estima quantas corridas podem ser realizadas e calcula a recompensa daquela decisão.</p>

```python
multiplicador = self.precos_multiplicadores[acao]
preco_base = 10.0
preco_final = preco_base * multiplicador

perda_elasticidade = int(
    max(0.0, (multiplicador - 1.0) * 15)
)
demanda_atendida = max(
    0, self.demanda - perda_elasticidade
)

corridas_realizadas = min(
    demanda_atendida, self.motoristas
)
receita = preco_final * corridas_realizadas

espera = max(
    0, demanda_atendida - self.motoristas
)
penalidade_espera = espera * 5.0

recompensa = receita - penalidade_espera
```

<p align="justify"><b>Preço final:</b> o preço de referência é de 10 unidades monetárias e é multiplicado pelo valor escolhido. Por exemplo, um multiplicador de 1,5 corresponderia a um preço de 15 unidades monetárias, caso esse multiplicador estivesse disponível na lista.</p>

<p align="justify"><b>Redução da procura:</b> quando o multiplicador supera 1,0, o código reduz a procura atendida por meio de uma regra simplificada de sensibilidade ao preço. Quanto maior o multiplicador, maior será essa redução simulada. Essa regra é uma hipótese do modelo, não uma estimativa obtida de dados reais de passageiros.</p>

<p align="justify"><b>Corridas realizadas e receita:</b> o número de corridas é limitado pelo menor valor entre a procura atendida e a quantidade de motoristas. A receita é calculada multiplicando o preço final pelo número de corridas realizadas.</p>

<p align="justify"><b>Penalidade por espera:</b> se a procura atendida superar a quantidade de motoristas, cada passageiro excedente gera uma penalidade de cinco unidades. O código também contém uma penalidade adicional para tarifas de até 1,0 em momentos de procura superior a 30. A recompensa final busca, portanto, equilibrar a receita e as penalidades definidas na simulação.</p>

<p align="justify"><b>Observação importante:</b> na implementação original, a penalidade adicional é calculada com <code>(self.demanda - self.motoristas) * 3.0</code>. Quando a procura é maior que 30, mas há mais motoristas do que passageiros, essa expressão pode ser negativa. Como a recompensa subtrai essa penalidade, o resultado pode se tornar um bônus inesperado. Para evitar isso, seria mais seguro limitar a diferença a zero com <code>max(0, self.demanda - self.motoristas)</code>.</p>

## 5. Passagem do tempo e encerramento do dia

<p align="justify">Depois de calcular a recompensa, o ambiente avança uma hora, sorteia novos valores de procura e de motoristas e atualiza o contador de etapas. Um episódio termina após 24 etapas, representando um dia completo. Essa ordem permite que o agente observe as condições atuais, escolha a tarifa e só então receba as condições da hora seguinte.</p>

```python
self.hora = (self.hora + 1) % 24
self.demanda = self._calcular_demanda_realista(self.hora)
self.motoristas = self._calcular_motoristas_realista(
    self.hora
)

self.tempo += 1
terminado = self.tempo >= 24
truncado = False

return (
    self._obter_observacao(),
    recompensa,
    terminado,
    truncado,
    {}
)
```

<p align="justify">O uso de <code>% 24</code> mantém o horário entre 0 e 23. A variável <code>terminado</code> informa quando o ciclo diário foi concluído. A variável <code>truncado</code> permanece falsa, pois o código não define outro limite de encerramento.</p>

## 6. Configuração e treinamento do agente

<p align="justify">O treinamento utiliza o método PPO, que ajusta gradualmente a política de decisão com base nas recompensas observadas. A configuração define o ambiente, o uso do PyTorch, a quantidade de dados reunidos por atualização, o tamanho dos grupos menores de treinamento, o número de passagens pelos dados e a taxa de aprendizado.</p>

```python
configuracao = (
    PPOConfig()
    .environment(AmbienteTarifaDinamica)
    .framework("torch")
    .env_runners(num_env_runners=0)
    .resources(num_gpus=0)
    .training(
        train_batch_size=1000,
        minibatch_size=100,
        num_epochs=2,
        lr=3e-4
    )
)

algoritmo = configuracao.build_algo()
historico_recompensas = []

for iteracao in range(20):
    resultado = algoritmo.train()
```

<p align="justify"><b>Quantidade de dados:</b> <code>train_batch_size=1000</code> define o volume de dados reunidos para uma atualização do treinamento. <code>minibatch_size=100</code> divide esse volume em grupos menores durante o ajuste.</p>

<p align="justify"><b>Passagens pelos dados:</b> <code>num_epochs=2</code> indica duas passagens de treinamento pelos dados reunidos em cada atualização, conforme a configuração utilizada.</p>

<p align="justify"><b>Taxa de aprendizado:</b> <code>lr=3e-4</code> controla o tamanho dos ajustes dos parâmetros durante o aprendizado. Uma taxa muito alta pode tornar o ajuste instável; uma taxa muito baixa pode exigir mais treinamento.</p>

<p align="justify"><b>Execução:</b> o laço realiza 20 iterações de treinamento e registra a recompensa média informada pela biblioteca. Vinte iterações servem como demonstração inicial, mas não garantem que a política tenha convergido para uma solução estável ou próxima da melhor possível.</p>

## 7. Acompanhamento do aprendizado

<p align="justify">O código guarda a recompensa média de cada iteração e constrói um gráfico para observar sua evolução. Uma tendência crescente pode indicar melhoria no comportamento do agente, mas oscilações são esperadas porque a procura e a quantidade de motoristas são sorteadas. Para avaliar o aprendizado com mais confiança, seria necessário repetir os experimentos com diferentes sementes aleatórias e comparar os resultados com estratégias simples, como manter sempre a tarifa de referência.</p>

```python
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, 21),
    historico_recompensas,
    marker="o",
    linestyle="-",
    linewidth=2
)
plt.title("Evolução do aprendizado do agente")
plt.xlabel("Iteração")
plt.ylabel("Recompensa média por episódio")
plt.xticks(range(1, 21))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```

<p align="justify">A recompensa média é uma medida do objetivo definido no ambiente, e não deve ser confundida automaticamente com lucro real. O cálculo atual considera receita e penalidades, mas não inclui custos operacionais, combustível, remuneração dos motoristas, impostos ou outros fatores financeiros.</p>

## 8. Teste da política aprendida

<p align="justify">Após o treinamento, o código obtém o módulo que representa a política aprendida e executa uma simulação de 24 horas. Em cada etapa, a observação é convertida em um tensor do PyTorch, o modelo calcula os valores associados às ações e o programa seleciona a ação de maior valor.</p>

```python
modulo_politica = algoritmo.get_module()

ambiente_teste = AmbienteTarifaDinamica()
observacao, _ = ambiente_teste.reset()

for passo in range(24):
    tensor_observacao = torch.from_numpy(
        np.expand_dims(observacao, axis=0)
    ).float()

    with torch.no_grad():
        entrada_modelo = {"obs": tensor_observacao}
        saida_modelo = modulo_politica.forward_inference(
            entrada_modelo
        )
        logits_acoes = saida_modelo["action_dist_inputs"]
        acao_escolhida = torch.argmax(
            logits_acoes, dim=-1
        ).item()

    observacao, recompensa, terminado, truncado, _ = (
        ambiente_teste.step(acao_escolhida)
    )
```

<p align="justify">O uso de <code>torch.no_grad()</code> indica que essa etapa é de avaliação, sem cálculo de gradientes para atualizar os parâmetros. A função <code>argmax</code> escolhe a ação com maior valor entre as saídas do modelo. Isso produz uma escolha determinística para a mesma observação e os mesmos parâmetros, embora as condições simuladas continuem variando aleatoriamente.</p>

<p align="justify">O laço imprime o horário, a procura, a quantidade de motoristas, o multiplicador aplicado e a recompensa. Esses dados permitem observar como o agente reage ao longo do dia. Para uma avaliação mais completa, também seria útil registrar a receita acumulada, o total de corridas, a espera acumulada e comparar a política aprendida com políticas de referência.</p>

## 9. Limitações e melhorias recomendadas

<p align="justify">A principal limitação é que a procura e a oferta de motoristas são geradas artificialmente por intervalos aleatórios. O ambiente reproduz um padrão diário simplificado, mas não foi ajustado com dados reais de uma cidade ou de uma plataforma de transporte. Portanto, os resultados demonstram o funcionamento do método, não comprovam que os preços calculados seriam adequados em uma operação real.</p>

<p align="justify">Outra limitação é a função de recompensa: ela usa receita como medida principal, sem descontar custos operacionais, e a regra de redução da procura é definida manualmente. Antes de aplicar o método a um serviço real, seria necessário estimar a resposta dos passageiros às tarifas, incorporar custos e restrições comerciais, corrigir a penalidade adicional e avaliar impactos sobre espera, cancelamentos e acessibilidade dos preços.</p>

<p align="justify">Também é recomendável fixar sementes aleatórias para tornar os testes reproduzíveis, treinar por mais iterações quando necessário e comparar o agente com regras simples de precificação. A avaliação deve usar vários dias simulados, em vez de depender de apenas uma execução de 24 horas.</p>

## 10. Conclusão

<p align="justify">O projeto apresenta uma estrutura inicial de precificação dinâmica com aprendizado por reforço. O ambiente representa a relação entre procura, disponibilidade de motoristas e tarifa; a função de recompensa define os incentivos; o treinamento ajusta a política; e a simulação final permite observar as decisões tomadas ao longo de um dia.</p>

<p align="justify">A contribuição principal é demonstrar como transformar uma decisão de preço em um problema de aprendizado sequencial. Para que a solução possa sustentar decisões comerciais, será necessário validar as hipóteses com dados reais, incluir custos e métricas operacionais, corrigir a penalidade que pode ficar negativa e realizar comparações experimentais reproduzíveis.</p>
