# LLMs for Political Decision-Making

## Research Question
How accurately can Large Language Models (LLMs) simulate human behaviour, particularly in decision-making? A study in the context of the US Senate.

## Abstract
Political scientists have long employed quantitative methods to model legislative decision-making, but recent advancements in Large Language Models (LLMs) have opened new possibilities for simulating complex political behavior. This study examines the potential of LLMs to replicate human decision-making in the context of the US Senate, utilizing a multi-agent framework to simulate the behavior of 100 LLM agents representing the 118th US Senate. The agents vote on real-world bills after engaging in three rounds of discussion, with simulated outcomes compared to actual legislative results to assess accuracy. By expanding upon the foundational work of Baker and Azher (2024), who first demonstrated the potential of LLMs for simulating believable government action, this research introduces a more complex simulation with a larger number of agents and focuses on vote outcomes rather than just text-based debate. The findings hold potential implications for the usage of autonomous agents in decision-making and the development of AI-driven simulations in social sciences.

## Introduction
Historically, legisltative processes and political preferences have been studied statistically since at least the 1920s, when a research article argued that the political vote can be represented as a frequency distribution of opinion (Rice 1924). Political scientists have increasingly employed quantitative predictive methods to model legislative processes in the US Congress, particularly to predict vote outcomes. These methods range from simple binary classifiers to more sophisticated models, such as logistic regression, text models and machine learning. One such logisitic regression model achieved 88% prediction accuracy without considering individual vote histories allowing it to be generalized to future new Congress members (Henighan and Kravit, 2015). Another study used word vectors to capture specific words that increase the probability of a bill becoming law and compared text models with context-only models (Nay 2017). Another study focused on social network analysis and introduced a heterogeneous graph to predict legislative votes based on legislator relationships and bill similarities (Wang, Varshney, and Aleksandra Mojsilović 2018).

With the advent of generative AI, the use of Large Language Models (LLMs) in forecasting is gaining increasing attention. A recent study demonstrated that an ensemble of LLMs could "achieve forecasting accuracy rivaling that of human crowd forecasting tournaments," effectively replicating the 'wisdom of the crowd' effect (Schoenegger et al. 2024).

This research tests a probabilistic approach to simulating political decision-making in the US senate using an LLM-powered multi-agent framework. The study constructs a network of 100 LLM agents that simulate the 118th US Senate, where the agents vote on real bills after three rounds of discussion. The simulated outcomes are tallied against the real world outcomes to get a measure of accuracy. 

This work offers contributions across several fields. In Political Science, it explores the potential of synthetic data for simulating survey studies, offers an alternative approach to modeling politics, and provides a sandbox for simulating social life by examining the quality and realism of LLM-generated data. In Computer Science, it contributes to understanding the predictive capabilities of LLMs and expanding their use in traditional machine learning tasks. In the field of Human-AI Co-evolution, it tests the idea of AI agents replicating human behaviour, helping us understand the potential for AI to interact with and complement human decision-making in real-world scenarios such as politics.

## Keywords
- **Computational politics:** "Computational Politics is the study of computational methods to analyze and moderate users' behaviors related to political activities" (Haq et al. 2019)
- **Agent:** An agent is software or hardware that can interact with its environment, collect data and use the data to autnomously perform tasks based on goals predetermined by a human. (“What Are AI Agents?- Agents in Artificial Intelligence Explained - AWS” 2024)
- **LLM:** Large Language Models or LLMs are a class of foundation models trained on massive datasets to generate human-like text. The underlying transformer is a set of neural networks. (“What Is LLM? - Large Language Models Explained - AWS” 2023)
- **Multiagent (MA) framework:** A multiagent system or framework consists of multiple AI agents, often LLM powered (referred to as LLM-MA), working collectively and interacting with each other in either a hierarchical or non-hierarchical manner to perform tasks. (Gutowska 2024)
- **Context window:** The context window or length is the amount of text (in tokens) that a LLM can remember at any one point. (IBM 2024)
- **Temperature:** Temperature is a parameter ranging from 0 to 1 affecting the variability and predictability of the generated output. A lower temperature value generates more deterministic output. A higher temperature value generates more creative and random output. ("LLM Settings" 2024)
- **System prompt:** This is the initial set of instructions that define the behaviour, role, goals and personality of a LLM.
- **Fine-tuning:** This involves additional training on examples specific to a task that refines the predictive capabilities of a model for that task. (Google for Developers 2025)

## Literature Review

In the rapidly evolving field of computational political science, December 2024 saw the introduction of the first comprehensive framework designed to integrate Large Language Models (LLMs) into the discipline. Developed by a team of multidisciplinary researchers, this framework, termed Political-LLM, represents a significant step forward in understanding how LLMs can be applied to political science research (Li et al. 2024). According to Political-LLM, applications of LLMs in political science for simulation can be divided into two primary categories: simulating behavior dynamics and facilitating text-based discussions.

In the first category, LLMs for simulating behavior dynamics, prior works have explored the potential of LLMs in modeling complex interactions and strategic behaviors. One example is the development of the self-evolving LLM-based diplomat, Richelieu (Guan 2024). Richelieu utilizes a multi-agent framework that incorporates roles such as planner, negotiator, and actor, along with a memory module for effective optimization. The study then employs another multi-agent framework to test Richelieu's performance by simulating the Diplomacy Game, involving seven agents representing different countries. The study concluded that Richelieu outperformed existing models, including Cicero, which was the first Diplomacy-playing LLM to achieve human-level performance (Meta 2022). 

Further research in this category examines the dynamics of conflict and cooperation in political settings. For instance, a study simulating an LLM-based "Artificial Leviathan" found that while initial interactions among agents were characterized by unrestrained conflict, over time, agents sought to escape their state of nature, forming social contracts and establishing a peaceful commonwealth (Dai 2024). In contrast, a study on behaviour of autonomous AI agents in high-stakes military and diplomatic decision-making concluded that "that all models show forms of escalation and difficult-to-predict escalation patterns that lead to greater conflict" (Rivera et. al 2024). Another study focused on the predictive capabilities of LLMs for social behaviour, investigating if a fine-tuned model could predict individual and aggregate policy preferences from a sample of 267 voters during Brazil’s 2022 presidential election. The LLM outperformed the traditional "bundle rule" model for out-of-sample voters, which assumes citizens vote according to the political orientation of their preferred candidate (Gudiño 2024).

The second category of LLMs for text-based discussions explores the use of LLMs in simulating political discourse. Notably, Baker and Azher (2024) observed that, prior to their study, no research had successfully simulated realistic government action using LLMs. Their work submitted for peer-review in June 2024 offered a proof-of-concept by simulating six AI senators in the 2024 US Senate Committee on Intelligence. In this simulation, the AI senators debated each other over three rounds on current issues (such as Russia’s invasion of Ukraine). The study found that domain experts considered the AI-generated interactions to be highly believable, and by introducing perturbations during the debate (such as "introduction of intelligence indicating imminent Russian overrun of Ukraine"), the researchers were able to identify shifts in decision-making and potential for bipartisanship. Similarly, another study introduces coalition negotiations across various European political parties as a novel NLP task by modeling them as negotiations between different LLM agents (Moghimifar 2024).

This paper builds on the research from both categories, testing the potential for the integration of LLMs into political science. Primarily, it utilizes demographic data to construct LLM agents that will simulate the behavior of US senators and their vote outcomes on different bills. It also places the agents in conversation with each other for text-based discussions, with the goal of adding a layer of realism in simulating the US Senate process and also identifying the changes (if any) that happen in decision-making through a multi-agent framework. It significantly expands on the work of Baker et al. (2024) by simulating an entire US senate (instead of just 6 senators) and focusing on vote outcomes (instead of just discussion).

## Background Knowledge
### Mechanisms of Generative AI and LLMs:
The training of Large Language Models (LLMs) follows a multi-step process:

1. Data Collection - The model is trained on massive text datasets sourced from books, articles, and websites to learn language patterns and contextual relationships.
2. Tokenization - The text is broken down into tokens, which are numerical representations that the model processes.
3. Pretraining - Using a neural network, typically a transformer architecture, the model undergoes unsupervised learning by predicting missing words or the next token in a sequence, adjusting billions of parameters to minimize prediction errors.
4. Fine-tuning - Some models undergo additional training on specialized datasets with supervised learning to improve performance on specific tasks.
5. Alignment - Reinforcement learning from human feedback (RLHF) is used to refine responses, ensuring relevance, coherence, and safety.

Once trained, LLMs make predictions by processing an input prompt, converting it into token embeddings, and passing these embeddings through multiple transformer layers. Each layer refines contextual understanding using mechanisms like self-attention, where the model assigns different weights to words based on their relevance to the context. The final layer generates probabilities for the next token, selecting the most likely sequence based on learned patterns, which is then decoded back into human-readable text. (Liu et al. 2024)

### LLMs for forecasting applications:
Forecasting methodologies are generally categorized into statistical and judgmental approaches. Statistical forecasting relies on quantitative data and mathematical models to predict future events. Techniques such as time series analysis, regression models, and econometric models are commonly used. On the other hand, judgmental forecasting involves subjective assessments and expert opinions to predict future events. This approach is often employed when historical data is limited, unreliable, or when forecasting unprecedented events. It leverages human intuition and experience, making it valuable in complex and uncertain environments. (Halawi et al. 2024)

  
  
### Is it approporiate to use LLMs for causal reasoning? (model collapse theory/generative AI paradox/hallucinations/LLMs do not understand what they are generating)
  
- How does political decision-making happen in the actual US Senate?
  

## Data
For this study, an original dataset was created by webscraping, collecting variables across two dimensions:
1. Demographic and ideological characteristics of the 118th US senators.
2. Contextual variables for 45 floor votes, including how each senator voted.

The final dataset has 4692 rows (one vote per senator per floor vote) and 51 columns.

### Data Sources
**For senators:**
- Congress.gov (https://www.congress.gov/): The official website for US federal legislative information. Utilized for basic biogrgaphical information (such as name, state, party, time served in house, time served in senate). 
- The Lugar Center (https://www.thelugarcenter.org/ourwork-Bipartisan-Index.html): Non-profit founded by former United States Senator Richard G. Lugar. Utilized for bipartisan index, "an objective measure of how well members of opposite parties work with one another using bill sponsorship and co-sponsorship data".
- Voteview (https://voteview.com/congress/senate/-1/text): Mainted by UCLA's Department of Political Science and Social Science Computing. Utilized for obtaining ideological position scores (calculated using the DW-NOMINATE).
- CQ Press Congress Collection (https://library-cqpress-com.proxy.lib.duke.edu/congress/): A research database containing biographical, political, and electoral data about every member of Congress since the 79th Congress (1945). Utilized for obtaining biographical data such as date of birth, education level, religion, race, sex; electoral data such as percentage of vote received in last election; political data such as frequency of voting, alignment with party positions, number of times senator voted with/against party, presidential support; as well as overall biography. Narrative biographies in the database are written by CQ editorial staff.
- The Cook Political Report (https://www.cookpolitical.com/cook-pvi): Introduced in 1997, the Cook Partisan Voting Index (Cook PVI) measures how each state and district performs at the presidential level compared to the nation as a whole. The dataset was utilized to get the 2023 Cook Partisan voting score for the state of each senator. The 2023 PVI scores were claculated using 2016 and 2020 presidential election results. 

**For floor votes:**

45 floor votes were selected to be simulated (votes included "On Motion to Discharge Committee", "On Overriding the Veto", "On Passage of the Bill", "On Cloture on the Motion to Proceed" etc.). These were randomly selected from a list containing CQ Key Votes ("for each series of related votes on an issue, only one vote is usually identified as a CQ Key Vote. This vote is the floor vote in the House or Senate that in the opinion of CQ's editorial staff was the most important in determining the outcome."), key legislations identified on Wikipedia and the most-viewed bills on Congress.gov. Bills originating in both the house and senate as well as joint resolutions were considered, although votes were limited to those for which there was a roll call vote in the senate.
- Congress.gov (https://www.congress.gov/): Utilized for obtaining contextual information about a bill such as title, summary, number of co-sponsors, party and name of introducing senator, policy area. Bill summaries are authored by the Congressional Research Service ("CRS provides Congress with analysis that is authoritative, confidential, objective, and non-partisan.")
- Senate.gov (https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_118_1.htm): Utilized for obtaining detailed records of how each senator voted during a particular floor vote (yea, nay, present or not voting).

### Target
The target variable being predicted is a senator's vote during a specific floor vote. This can take 4 values: Yea, Nay, Present or Not Voting. 

### Features
The features fed into were: 

### Data Wrangling
The target variable 
Target Codified as 1, 0 for basic EDA, filtered not voting.
Categorical variables coded
**Missing variables:** The bipartisan index for the senate majority and minority leaders was missing. Nebraska and Maine had different PVI scores for their different electoral seats (the average was taken). Any missing biographical and electoral data in the CQ Press Congress Collection was manually filled by verifying through news reports.

### Variable Codebook
Variable, type, description, source
The Bipartisan Index measures the frequency with which a Member co-sponsors a bill introduced by the opposite party and the frequency with which a Member’s own bills attract co-sponsors from the opposite party. Measured on a scale of

## Research Design
### Overview

### Selection of simulation context
### Selection of LLM model and memory-handling
### Selection of multi-agent framework
### Determining the order in which agents will interact
### Controlling model parameters (temperature, system prompt cut-off date)

## Methodology
### Exploratory Data Analysis (correlation matrix and statistical models)
### Basic Simulation (without prompt engineering and without controlling any parameters)
### Vote simulation with prompt engineering, with controlling parameters
### Vote simulation with prompt engineering, with controlling parameters and after 3 rounds of debate

## An AI Senator
This is what an LLM-powered senator looks like:
![AI_Agent_Slide-1](https://github.com/user-attachments/assets/7cc77339-7f24-448d-83b2-25c937e1ff9a)

## A Sample Simulation
<img src="https://github.com/user-attachments/assets/09cefb2e-1701-4434-8a83-694719d98f03" width="600"/>
<img src="https://github.com/user-attachments/assets/03962bfa-d0c1-4c08-ad24-a2223d0215a4" width="600"/>
<img src="https://github.com/user-attachments/assets/15400582-18d9-4a41-af65-104856f574ff" width="600"/>

## Results

## File Description
This github repository contains all the code and data utilized to generate results for this study.

## References
- Henighan, Tom, and Scott Kravitz. 2015. “Predicting Bill Votes in the House of Representatives.” December 12, 2015. https://tomhenighan.com/pdfs/vote_prediction.pdf.
- Nay, John J. 2017. “Predicting and Understanding Law-Making with Word Vectors and an Ensemble Model.” PLoS ONE 12 (5): e0176999–99. https://doi.org/10.1371/journal.pone.0176999.
- Schoenegger, Philipp, Indre Tuminauskaite, Peter S Park, and Philip E Tetlock. 2024. “Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy.” ArXiv.org. 2024. https://arxiv.org/abs/2402.19379.
- Li, Lincan, Jiaqi Li, Catherine Chen, Fred Gui, Hongjia Yang, Chenxiao Yu, Zhengguang Wang, et al. 2024. “Political-LLM: Large Language Models in Political Science.” ArXiv.org. 2024. https://arxiv.org/abs/2412.06864.
- Guan, Zhenyu, Xiangyu Kong, Fangwei Zhong, and Yizhou Wang. 2024. “Richelieu: Self-Evolving LLM-Based Agents for AI Diplomacy.” ArXiv.org. 2024. https://arxiv.org/abs/2407.06813.
- Dai, Gordon, Weijia Zhang, Jinhan Li, Siqi Yang, lbe, Chidera Onochie, Srihas Rao, Arthur Caetano, and Misha Sra. 2024. “Artificial Leviathan: Exploring Social Evolution of LLM Agents through the Lens of Hobbesian Social Contract Theory.” ArXiv.org. 2024. https://arxiv.org/abs/2406.14373.
- Gudiño, Jairo F, Umberto Grandi, and César Hidalgo. 2024. “Large Language Models (LLMs) as Agents for Augmented Democracy.” Philosophical Transactions of the Royal Society a Mathematical Physical and Engineering Sciences 382 (2285). https://doi.org/10.1098/rsta.2024.0100.
- Moghimifar, Farhad, Yuan-Fang Li, Robert Thomson, and Gholamreza Haffari. 2024. “Modelling Political Coalition Negotiations Using LLM-Based Agents.” ArXiv.org. 2024. https://arxiv.org/abs/2402.11712.
- Baker, Zachary R, and Azher, Zarif L. 2024. “Simulating the U.S. Senate: An LLM-Driven Agent Approach to Modeling Legislative Behavior and Bipartisanship.” ArXiv.org. 2024. https://arxiv.org/abs/2406.18702.
- “Escalation Risks from LLMs in Military and Diplomatic Contexts | Stanford HAI.” 2019. Stanford.edu. 2019. https://hai.stanford.edu/policy/policy-brief-escalation-risks-llms-military-and-diplomatic-contexts.
- Wang, Jun, Kush R Varshney, and Aleksandra Mojsilović. 2018. “Legislative Prediction with Political and Social Network Analysis.” Springer EBooks, January, 1184–91. https://doi.org/10.1007/978-1-4939-7131-2_285.
- Rice, Stuart A. 1924. “The Political Vote as a Frequency Distribution of Opinion.” Journal of the American Statistical Association 19 (145): 70–75. doi:10.1080/01621459.1924.10502872.
- “CICERO.” 2022. Meta.com. 2022. https://ai.meta.com/research/cicero/.
- Haq, Ehsan ul, Tristan Braud, Young D Kwon, and Pan Hui. 2019. “A Survey on Computational Politics.” ArXiv.org. 2019. https://arxiv.org/abs/1908.06069.
- “What Are AI Agents?- Agents in Artificial Intelligence Explained - AWS.” 2024. Amazon Web Services, Inc. 2024. https://aws.amazon.com/what-is/ai-agents/.
- Gutowska, Anna. 2024. “Multiagent System.” Ibm.com. August 6, 2024. https://www.ibm.com/think/topics/multiagent-system.
- “What Is LLM? - Large Language Models Explained - AWS.” 2023. Amazon Web Services, Inc. 2023. https://aws.amazon.com/what-is/large-language-model/
- IBM. 2024. “Context Window.” Ibm.com. November 7, 2024. https://www.ibm.com/think/topics/context-window.
- “LLM Settings” 2024. Promptingguide.ai. 2024. https://www.promptingguide.ai/introduction/settings.
- Google for Developers. 2025. “LLMs: Fine-Tuning, Distillation, and Prompt Engineering.” developers.google.com. 2025. https://developers.google.com/machine-learning/crash-course/llm/tuning.
- Liu, Yiheng, Hao He, Tianle Han, Xu Zhang, Mengyuan Liu, Jiaming Tian, Yutong Zhang, et al. 2024. “Understanding LLMs: A Comprehensive Overview from Training to Inference.” ArXiv.org. 2024. https://arxiv.org/abs/2401.02038.
- Halawi, Danny, Fred Zhang, Chen Yueh-Han, and Jacob Steinhardt. 2024. “Approaching Human-Level Forecasting with Language Models.” ArXiv.org. 2024. https://arxiv.org/abs/2402.18563.
‌
