# LLMs for Political Decision-Making

## Research Question
How accurately can Large Language Models (LLMs) simulate human behaviour, particularly in decision-making? A study in the context of the US Senate.

## Introduction
Political scientists have increasingly employed quantitative predictive methods to model legislative processes in the US Congress, particularly to predict vote outcomes. These methods range from simple binary classifiers to more sophisticated models, such as logistic regression, text models and machine learning. One such logisitic regression model achieved 88% prediction accuracy without considering individual vote histories allowing it to be generalized to future new Congress members (Henighan & Kravitz, 2015). Another study used word vectors to capture specific words that increase the probability of a bill becoming law and compared text models with context-only models (Nay, 2017). Another study focused on social network analysis and introduced a heterogeneous graph to predict legislative votes based on legislator relationships and bill similarities (Wang et. al 2018).

With the advent of generative AI, the use of Large Language Models (LLMs) in forecasting is gaining increasing attention. A recent study demonstrated that an ensemble of LLMs could "achieve forecasting accuracy rivaling that of human crowd forecasting tournaments," effectively replicating the 'wisdom of the crowd' effect (Schoenegger et al., 2024).

This research tests a probabilistic approach to simulating political decision-making in the US senate using an LLM-powered multi-agent framework. The study constructs a network of 100 LLM agents that simulate the 118th US Senate, where the agents vote on real bills after three rounds of discussion. The simulated outcomes are tallied against the real world outcomes to get a measure of accuracy. 

This work offers contributions across several fields. In Political Science, it explores the potential of synthetic data for simulating survey studies, offers an alternative approach to modeling politics, and provides a sandbox for simulating social life by examining the quality and realism of LLM-generated data. In Computer Science, it contributes to understanding the predictive capabilities of LLMs and expanding their use in traditional machine learning tasks. In the field of Human-AI Co-evolution, it tests the idea of AI agents replicating human behaviour, helping us understand the potential for AI to interact with and complement human decision-making in real-world scenarios such as politics.

## Keywords
- Computational political science
- Agent
- LLM
- Multi-agent framework
- Memory stream
- Wisdom of the crowd
- Fine-tuning

## Literature Review

In the rapidly evolving field of computational political science, December 2024 saw the introduction of the first comprehensive framework designed to integrate Large Language Models (LLMs) into the discipline. Developed by a team of multidisciplinary researchers, this framework, termed Political-LLM, represents a significant step forward in understanding how LLMs can be applied to political science research (Li et al., 2024). According to Political-LLM, applications of LLMs in political science can be divided into two primary categories: simulating behavior dynamics and facilitating text-based discussions.

In the first category, LLMs for simulating behavior dynamics, prior works have explored the potential of LLMs in modeling complex interactions and strategic behaviors. One example is the development of the self-evolving LLM-based diplomat, Richelieu (Guan, 2024). Richelieu utilizes a multi-agent framework that incorporates roles such as planner, negotiator, and actor, along with a memory module for effective optimization. The study then employs another multi-agent framework to test Richelieu's performance by simulating the Diplomacy Game, involving seven agents representing different countries. The study concluded that Richelieu outperformed existing models, including Cicero, which was the first Diplomacy-playing LLM to achieve human-level performance (Meta, 2024). 

Further research in this category examines the dynamics of conflict and cooperation in political settings. For instance, a study simulating an LLM-based "Artificial Leviathan" found that while initial interactions among agents were characterized by unrestrained conflict, over time, agents sought to escape their state of nature, forming social contracts and establishing a peaceful commonwealth (Dai, 2024). In contrast, a study on behaviour of autonomous AI agents in high-stakes military and diplomatic decision-making concluded that "that all models show forms of escalation and difficult-to-predict escalation patterns that lead to greater conflict" (Rivera et. al 2024). Another study focused on the predictive capabilities of LLMs for social behaviour, investigating if a fine-tuned model could predict individual and aggregate policy preferences from a sample of 267 voters during Brazil’s 2022 presidential election. The LLM outperformed the traditional "bundle rule" model for out-of-sample voters, which assumes citizens vote according to the political orientation of their preferred candidate (Gudiño, 2024).

The second category of LLMs for text-based discussions explores the use of LLMs in simulating political discourse. Notably, Baker et al. (2024) observed that, prior to their study, no research had successfully simulated realistic government action using LLMs. Their work submitted for peer-review in June 2024 offered a proof-of-concept by simulating six AI senators in the 2024 US Senate Committee on Intelligence. In this simulation, the AI senators debated each other over three rounds on current issues (such as Russia’s invasion of Ukraine). The study found that domain experts considered the AI-generated interactions to be highly believable, and by introducing perturbations during the debate (such as "introduction of intelligence indicating imminent Russian overrun of Ukraine"), the researchers were able to identify shifts in decision-making and potential for bipartisanship. Similarly, another study introduces coalition negotiations across various European political parties as a novel NLP task by modeling them as negotiations between different LLM agents (Moghimifar 2024).

This paper builds on the research from both categories, testing the potential for the integration of LLMs into political science. Primarily, it utilizes demographic data to construct LLM agents that will simulate the behavior of US senators and their vote outcomes on different bills. It also places the agents in conversation with each other for text-based discussions, with the goal of adding a layer of realism in simulating the US Senate process and also identifying the changes (if any) that happen in decision-making through a multi-agent framework. It significantly expands on the work of Baker et al. (2024) by simulating an entire US senate (instead of just 6 senators) and focusing on vote outcomes (instead of just discussion).

## Context
- How does generative AI make predictions? (review on studies using LLMs for forecasting)
- Is it approporiate to use LLMs for causal reasoning? (model collapse theory)
- How does political decision-making happen in the actual US Senate?

## Data
For this study, an original dataset was created by webscraping, collecting variables across two dimensions:
1. Demographic and ideological characteristics of the 118th US senators.
2. Contextual variables for 50 floor votes, including how each senator voted.

The final dataset has 500 rows (one vote per senator per floor vote) and 49 columns.

### Data Sources
**For senators:**
- Congress.gov (https://www.congress.gov/): The official website for US federal legislative information. Utilized for basic biogrgaphical information (such as name, state, party, time served in house, time served in senate). 
- The Lugar Center (https://www.thelugarcenter.org/ourwork-Bipartisan-Index.html): Non-profit founded by former United States Senator Richard G. Lugar. Utilized for bipartisan index, "an objective measure of how well members of opposite parties work with one another using bill sponsorship and co-sponsorship data".
- Voteview (https://voteview.com/congress/senate/-1/text): Mainted by UCLA's Department of Political Science and Social Science Computing. Utilized for obtaining ideological position scores (calculated using the DW-NOMINATE).
- CQ Press Congress Collection (https://library-cqpress-com.proxy.lib.duke.edu/congress/): A research database containing biographical, political, and electoral data about every member of Congress since the 79th Congress (1945). Utilized for obtaining biographical data such as date of birth, education level, religion, race, sex; electoral data such as percentage of vote received in last election; political data such as frequency of voting, alignment with party positions, number of times senator voted with/against party, presidential support; as well as overall biography. Narrative biographies in the database are written by CQ editorial staff.
- The Cook Political Report (https://www.cookpolitical.com/cook-pvi): Introduced in 1997, the Cook Partisan Voting Index (Cook PVI) measures how each state and district performs at the presidential level compared to the nation as a whole. The dataset was utilized to get the 2023 Cook Partisan voting score for the state of each senator. The 2023 PVI scores were claculated using 2016 and 2020 presidential election results. 

**For floor votes:**

50 floor votes were selected to be simulated (including roll call votes, voice votes and colture motions). These were selected based on the CQ Key Votes ("for each series of related votes on an issue, only one vote is usually identified as a CQ Key Vote. This vote is the floor vote in the House or Senate that in the opinion of CQ's editorial staff was the most important in determining the outcome."), key legislations identified on Wikipedia and the most-viewed bills identified on Congress.gov. Only bills originating in the senate were considered.
- Congress.gov (https://www.congress.gov/): Utilized for obtaining contextual information about a bill such as title, summary, number of co-sponsors, party and name of introducing senator, policy area. Bill summaries are authored by the Congressional Research Service ("CRS provides Congress with analysis that is authoritative, confidential, objective, and non-partisan.")
- Senate.gov (https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_118_1.htm): Utilized for obtaining detailed records of how each senator voted on a particular bill (yea, nay, not voting).

### Target
Vote on bill (Yea, Nay, Not Voting). Codified as 1, 0 for basic EDA, filtered not voting.

### Features

### Data Wrangling
Target Codified as 1, 0 for basic EDA, filtered not voting.
Categorical variables coded
(how missing variables were filled)

### Variable Codebook
Variable, type, description, source
The Bipartisan Index measures the frequency with which a Member co-sponsors a bill introduced by the opposite party and the frequency with which a Member’s own bills attract co-sponsors from the opposite party. Measured on a scale of

## Methodology

## Results

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
- https://arxiv.org/abs/2406.18702
- https://hai.stanford.edu/policy/policy-brief-escalation-risks-llms-military-and-diplomatic-contexts
- https://doi.org/10.1007/978-1-4939-7131-2_285
