# LLMs for Political Decision-Making

## Research Question
How accurately can Large Language Models (LLMs) simulate human behaviour, particularly in decision-making? A study in the context of the US Senate.

## Introduction
Political scientists have increasingly employed quantitative predictive methods to model legislative processes in the US Congress, particularly to predict vote outcomes. These methods range from simple binary classifiers to more sophisticated models, such as logistic regression, text models and machine learning. One such logisitic regression model achieved 88% prediction accuracy without considering individual vote histories allowing it to be generalized to future new Congress members (Henighan & Kravitz, 2015). Another study used word vectors to capture specific words that increase the probability of a bill becoming law and compared text models with context-only models (Nay, 2017).

With the advent of generative AI, the use of Large Language Models (LLMs) in forecasting is gaining increasing attention. A recent study demonstrated that an ensemble of LLMs could "achieve forecasting accuracy rivaling that of human crowd forecasting tournaments," effectively replicating the 'wisdom of the crowd' effect (Schoenegger et al., 2024).

This research tests a probabilistic approach to simulating political decision-making in the US senate using an LLM-powered multi-agent framework. The study constructs a network of 100 LLM agents that simulate the 118th US Senate, where the agents vote on real bills after three rounds of discussion. The simulated outcomes are tallied against the real world outcomes to get a measure of accuracy. 

This work offers contributions across several fields. In Political Science, it explores the potential of synthetic data for simulating survey studies, offers an alternative approach to modeling politics, and provides a sandbox for simulating social life by examining the quality and realism of LLM-generated data. In Computer Science, it contributes to understanding the predictive capabilities of LLMs and expanding their use in traditional machine learning tasks. In the field of Human-AI Co-evolution, it tests the idea of AI agents replicating human behaviour, helping us understand the potential for AI to interact with and complement human decision-making in real-world scenarios such as politics.

## Keywords
- Computational political science
- Agent
- LLM
- Multi-agent framework
- Wisdom of the crowd
- Fine-tuning

## Literature Review

With computational political science being a rapidly advancing field, in December 2024 for the first time a team of multidisciplinary researchers developed "the first principled framework termed Political-LLM to advance the comprehensive understanding of integrating LLMs into computational political science" (Li et. al, 2024). According to Political-LLM, "applications of simulation agents [in Political Science] can be divided into two categories: simulating behavior dynamics and text-based discussions." 

In the first category (LLMs for simulating behavior dynamics), we see prior work such as the development of the self-evolving LLM-based diplomat, Richelieu (Guan, 2024). Richelieu employs a multi-agent framework across the roles of planner, negotiator and actor with "a memory module for effective optimization". The study then utilizes a multi-agent framework to simulate the Diplomacy Game with seven agents representing seven different countries, concluding that Richelieu ultimately outperforms "existing models like Cicero in the Diplomacy" (the first Diplomacy-playing LLM to achieve human-level performance in the game, built by Meta).

Another work in this category simulates an LLM-based "Artificial Leviathan" finding that though initially agents engage in unrestrained conflict, they seek to escape their brutish state of nature and eventually social contracts are formed and a peaceful commenwealth emergres based on those contracts (Dai 2024). Yet another study explores the ability of a fine-tuned LLM to predict the individual and aggregate policy preferences of 267 voters during Brazil’s 2022 presidential election, finding that at the individual level LLMs predict out of sample preferences more accurately than a ‘bundle rule’, which would assume that citizens always vote for the proposals of the candidate aligned with their self-reported political orientation (Gudiño 2024).

In the second category (LLMs for text-based discussions), . Another study utilizes agents to model coalition negotiations across different European political parties (Moghimifar 2024). 


## Context
- How does generative AI make predictions? Is it approporiate to use LLMs for causal reasoning?
- How does political decision-making happen in the actual US Senate?

## Data

## Methodology

## File Description

## References

- https://tomhenighan.com/pdfs/vote_prediction.pdf
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5425031/
- https://arxiv.org/abs/2402.19379
- https://political-llm.org/
- https://arxiv.org/html/2407.06813v1#S4
- https://arxiv.org/abs/2406.14373
- https://royalsocietypublishing.org/doi/10.1098/rsta.2024.0100
- https://arxiv.org/abs/2402.11712
