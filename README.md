# Recursive-file-exploration-RAG

Repositório do trabalho de conclusão de curso "Abordagem alternativa de RAG: Recuperação iterativa de contexto com agentes inteligentes" - do curso de Ciência de Dados e Inteligência Artificial da FGV EMAp.

Resumo do trabalho:
> A proposta desse trabalho é trazer uma abordagem alternativa para o campo de recuperação de contexto para modelos de geração aumentada por recuperação (Retrieval-Augmented Generation - RAG). O objetivo principal foi desenvolver uma alternativa ao RAG tradicional, substituindo a busca semântica direta por um agente inteligente capaz de realizar curadoria iterativa de informações em bases de conhecimento heterogêneas. Essa abordagem prioriza a qualidade e a precisão das respostas em cenários de pequeno e médio porte, onde a integração e exploração de dados fragmentados pode ser realizada de forma mais robusta.
>
> A metodologia baseou-se na implementação de um agente que utiliza o framework LangGraph para modelar o fluxo de execução como um grafo. O agente constrói progressivamente o contexto necessário para responder perguntas, explorando arquivos de diversos formatos, como textos, imagens, tabelas e notebooks. A exploração é orientada por estratégias iterativas de busca, integrando dados de maneira incremental e utilizando técnicas de prompt engineering para refinar o contexto.
>
> Os testes comparativos foram realizados em bases de conhecimento simuladas, e o sistema demonstrou ser uma alternativa promissora para modelos RAG em aplicações específicas, contribuindo para o avanço de soluções de inteligência artificial mais adaptáveis e precisas.

## Estrutura do repositório

#### Bases de conhecimento simuladas utilizadas para experimentos:
- 📂 [Base de Conhecimento 1](./bases/base_conhecimento_1)
    > Contem arquivos de texto com informações fragmentadas em multiplos arquivos.
    
    📄 [resumo da base de conhecimento 1.md](./resumo%20da%20base%20de%20conhecimento%201.md)
- 📂 [Base de Conhecimento 2](./bases/base_conhecimento_2)
    > Contem arquivos de texto, dados, imagem e notebooks com informações fragmentadas em multiplos arquivos.

    📄 [resumo da base de conhecimento 2.md](./resumo%20da%20base%20de%20conhecimento%202.md)

- 📂 [prompts](./prompts/)
    > Contem templates de prompts utilizados para instruir os agentes.

- 📂 [respostas](./respostas/)
    > Contem respostas geradas pelos agentes nas perguntas de teste

- 📄 [file_integration.py](./file_interaction.py)
    > Contem o codigo fonte para os agentes que realizam a extração de informações dos arquivos.

- 📄 [recursive_file_exploration_rag.py](./recursive_file_exploration_rag.py)
    > Contem o codigo fonte para a aplicação de respostas com recuperação iterativa de contexto.

- 📄[recursive_file_explorer_rag.ipynb](./recursive_file_explorer_rag.ipynb)
    > Notebook com exemplo de execução do agente.

- 📄 [rfe_rag_evaluation.ipynb](./rfe_rag_evaluation.ipynb)
    > Notebook contendo as perguntas e o código de avaliação do agente.

- 📄 [setup_inicial.ipynb](./setup_inicial.ipynb)
    > Notebook com com as instalações das bibiliotecas necessárias.

- 📄 [testando_recuperacao_em_tabelas_e_notebooks.ipynb](./testando_recuperacao_em_tabelas_e_notebooks.ipynb)
    > Notebook com exemplos de recuperação de informações em tabelas e notebooks.

- 📄 [utils.py](./utils.py)
    > Contem funções utilitárias para a execução dos agentes, e também algumas definições gerais, como o objeto de estado interno da aplicação ou algumas estruturas de resposta.