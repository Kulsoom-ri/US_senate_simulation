# LLMs for Political Decision-Making

## Research Question
How accurately can Large Language Models (LLMs) simulate human behaviour, particularly in decision-making? A study in the context of the US Senate.

## Abstract
Political scientists have long employed quantitative methods to model legislative decision-making, but recent advancements in Large Language Models (LLMs) have opened new possibilities for simulating complex political behavior. This study examines the potential of LLMs to replicate human decision-making in the context of the US Senate, utilizing a multi-agent framework to simulate the behavior of 100 LLM agents representing the 118th US Senate. The agents vote on real-world bills after engaging in two rounds of discussion, with simulated outcomes compared to actual legislative results to assess accuracy. By expanding upon the foundational work of Baker and Azher (2024), who first demonstrated the potential of LLMs for simulating believable government action, this research introduces a more complex simulation with a larger number of agents and focuses on vote outcomes rather than just text-based debate. The findings hold potential implications for the usage of autonomous agents in decision-making and the development of AI-driven simulations in social sciences.

## Introduction
Historically, legislative processes and political preferences have been studied statistically since at least the 1920s, when a research article argued that the political vote can be represented as a frequency distribution of opinion (Rice 1924). Political scientists have increasingly employed quantitative predictive methods to model legislative processes in the US Congress, particularly to predict vote outcomes. These methods range from simple binary classifiers to more sophisticated models, such as logistic regression, text models and machine learning. One such logistic regression model achieved 88% prediction accuracy without considering individual vote histories allowing it to be generalized to future new Congress members (Henighan and Kravit, 2015). Another study used word vectors to capture specific words that increase the probability of a bill becoming law and compared text models with context-only models (Nay 2017). Another study focused on social network analysis and introduced a heterogeneous graph to predict legislative votes based on legislator relationships and bill similarities (Wang, Varshney, and Aleksandra Mojsilović 2018).

With the advent of generative AI, the use of Large Language Models (LLMs) in forecasting is gaining increasing attention. A recent study demonstrated that an ensemble of LLMs could "achieve forecasting accuracy rivaling that of human crowd forecasting tournaments," effectively replicating the 'wisdom of the crowd' effect (Schoenegger et al. 2024).

This research tests a probabilistic approach to simulating political decision-making in the US senate using an LLM-powered multi-agent framework. The study constructs a network of 100 LLM agents that simulate the 118th US Senate, where the agents vote on real bills after two rounds of discussion. The simulated outcomes are tallied against the real world outcomes to get a measure of accuracy. 

This work offers contributions across several fields. In Political Science, it explores the potential of synthetic data for simulating survey studies, offers an alternative approach to modeling politics, and provides a sandbox for simulating social life by examining the quality and realism of LLM-generated data. In Computer Science, it contributes to understanding the predictive capabilities of LLMs and expanding their use in traditional machine learning tasks. In the field of Human-AI Co-evolution, it tests the idea of AI agents replicating human behaviour, helping us understand the potential for AI to interact with and complement human decision-making in real-world scenarios such as politics.

## Keywords
- **Computational politics:** "Computational Politics is the study of computational methods to analyze and moderate users' behaviors related to political activities" (Haq et al. 2019)
- **Agent:** An agent is software or hardware that can interact with its environment, collect data and use the data to autonomously perform tasks based on goals predetermined by a human. (“What Are AI Agents?- Agents in Artificial Intelligence Explained - AWS” 2024)
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

This paper builds on the research from both categories, testing the potential for the integration of LLMs into political science. Primarily, it utilizes demographic data to construct LLM agents that will simulate the behavior of US senators and their vote outcomes on different bills. It also places the agents in conversation with each other for text-based discussions, with the goal of adding a layer of realism in simulating the US Senate process and also identifying the changes (if any) that happen in decision-making through a multi-agent framework. It significantly expands on the work of Baker and Azher (2024) by simulating an entire US senate (instead of just 6 senators) and focusing on vote outcomes (instead of just discussion).

## Background Knowledge
### How do LLMs work?
The training of Large Language Models (LLMs) follows a multi-step process:

1. Data Collection - The model is trained on massive text datasets sourced from books, articles, and websites to learn language patterns and contextual relationships.
2. Tokenization - The text is broken down into tokens, which are numerical representations that the model processes.
3. Pretraining - Using a neural network, typically a transformer architecture, the model undergoes unsupervised learning by predicting missing words or the next token in a sequence, adjusting billions of parameters to minimize prediction errors.
4. Fine-tuning - Some models undergo additional training on specialized datasets with supervised learning to improve performance on specific tasks.
5. Alignment - Reinforcement learning from human feedback (RLHF) is used to refine responses, ensuring relevance, coherence, and safety.

Once trained, LLMs make predictions by processing an input prompt, converting it into token embeddings, and passing these embeddings through multiple transformer layers. Each layer refines contextual understanding using mechanisms like self-attention, where the model assigns different weights to words based on their relevance to the context. The final layer generates probabilities for the next token, selecting the most likely sequence based on learned patterns, which is then decoded back into human-readable text. (Liu et al. 2024)

### Can LLMs be used for forecasting?
Forecasting methodologies are generally categorized into statistical and judgmental approaches. Statistical forecasting relies on quantitative data and mathematical models to predict future events. Techniques such as time series analysis, regression models, and econometric models are commonly used. On the other hand, judgmental forecasting involves subjective assessments and expert opinions to predict future events. This approach is often employed when historical data is limited, unreliable, or when forecasting unprecedented events. It leverages human intuition and experience, making it valuable in complex and uncertain environments. (Halawi et al. 2024)

However, judgmental forecasting is time- and labor-intensive, prompting interest in automating the process using large language models (LLMs) since they are already trained on vast amounts of cross-domain data. LLMs forecast by integrating diverse textual data, reasoning through context, and generating probabilistic predictions, in comparison to traditional ML classifiers that rely on structured data. Halawi et al. (2024) demonstrate that a retrieval-augmented LLM-based forecasting system achieves a Brier score of 0.179 and an accuracy of 71.5%, coming close to the human crowd’s 0.149 Brier score and 77.0% accuracy (Brier scores are a measure of the accuracy of probabilistic predictions, calculated as the mean squared difference between a predicted probability and the actual outcome). The system excels when the crowd is uncertain and when more relevant articles are retrieved but struggles with over-hedging in high-certainty cases.

Similarly, across various disciplines studies have tested the predictive power of LLMs on diverse datasets, revealing both their strengths and limitations. In neuroscience, a study evaluated general-purpose LLMs against expert neuroscientists on BenchBrain, a benchmark designed for predicting neuroscience research outcomes. The general-purpose LLMs achieved 81% accuracy in predicting experimental outcomes and the fine-tuned model (BrainGPT) performed even better (Luo et al. 2024). Another study explored whether LLMs trained for language prediction could effectively forecast multivariate time series data, showing preliminary but definitive success in demand forecasting tasks, albeit with concerns about overfitting (Wolff et al. 2025).

Another study assessed LLMs on theory of mind (ToM) tasks i.e. evaluations designed to assess an individual's ability to understand and attribute mental states (like beliefs, desires, and intentions) to others. Eleven LLMs were tested on 40 false-belief tasks, a gold-standard benchmark for evaluating ToM capabilities. While older models failed entirely, GPT-3.5-turbo achieved 20% accuracy, and GPT-4 reached 75%, performing at the level of a six-year-old child. These results suggest that ToM-like reasoning may have emerged as an unintended byproduct of language model training (Kosinski 2024). This has significant implications for forecasting and decision-making, as improved reasoning capabilities could enhance LLMs’ ability to predict human behavior, market trends, and social dynamics. However, the study also underscores limitations- LLMs may still struggle with nuanced, open-ended reasoning, meaning their forecasts could lack depth or misinterpret complex scenarios. Moreover, if these capabilities emerge unintentionally, it raises concerns about unpredictability in decision-making applications. While LLMs show promise in assisting with forecasting, their results must be interpreted with caution.
  
### Can LLMs provide explainable predictions, and are they appropriate for causal reasoning?
Political-LLM notes that explanability of LLM outputs is essential as it ensures that results are interpretable and transparent, "fostering trust in politically sensitive applications". LLMs potentially offer novel tools for causal analysis by identifying patterns, modeling causal relationships, and generating counterfactual scenarios to explore "what-if" conditions. Explainability tools, including attention mechanisms, hypertuning parameters and prompt engineering, could enhance the transparency of LLM-driven causal analysis.

Despite these strengths, LLMs face limitations that challenge their reliability for causal reasoning. A study from 2023 presents the 'Generative AI Paradox' which states that "generative models seem to acquire generation abilities more effectively than understanding, in contrast to human intelligence where generation is usually harder." (West et al. 2023) The study tests whether language models (such as GPT4) truly understand their own generated content by asking them multiple-choice questions about it. The LLMs perform worse than humans in discerning their own generations as well as in answering questions about them (although only slightly worse on the latter with an average accuracy of almost 87%).

Model collapse, a degenerative process where models gradually lose touch with the true underlying data distribution when trained on AI-generated content, poses another challenge for predictive multi-agent frameworks (Shumailov et al. 2024). In such systems, where one model’s output serves as another’s input, errors and biases can compound over time, leading to degraded performance. This issue arises because generative models, without regular exposure to diverse, high-quality human-generated data, increasingly reinforce their own distortions rather than accurately reflecting real-world distributions. Addressing model collapse requires a combination of fine-tuning, data filtering, and reinforcement learning techniques to maintain model integrity and prevent systemic degradation.

Finally, the issue of hallucinations and the "Black Box" problem in LLMs poses a significant limitation for any predictive task. One study positions hallucinations as a structural issue by proving rigorously that: (1) no training dataset can achieve 100% completeness, thus guaranteeing the model will encounter unknown or contradictory information; (2) the "needle in a haystack" problem, or accurate information retrieval, is mathematically undecidable, meaning LLMs cannot deterministically locate correct facts within their data; (3) intent classification is also undecidable, preventing accurate interpretation of user prompts; (4) the LLM halting problem is undecidable, rendering the length and content of generated outputs unpredictable; and (5) even with fact-checking, the inherent limitations of LLMs prevent complete elimination of hallucinations. (Banerjee, Agarwal, and Singla 2024) The "Black Box" problem presents a significant challenge, as comprehending the internal mechanisms of LLMs remains difficult even when they provide explanations for their outputs. Additionally, the lack of standardized evaluation benchmarks- specifically, the absence of a universally accepted metric for measuring explainability- limits the assessment of these models. Despite these challenges, the field of LLM explainability is rapidly evolving, presenting opportunities for applying them in different research contexts.

### How does political decision-making happen in the actual US Senate?
Despite the challenges of applying LLMs to forecasting tasks, rapid advancements in model development- such as expanded memory context windows, improved reasoning capabilities, and larger pre-training datasets- are enhancing their potential. Techniques like Retrieval-Augmented Generation (RAG) further refine their ability to access and synthesize relevant information. This paper thus explores the application of LLMs to social science simulations, specifically modeling decision-making in the U.S. Senate. To accurately simulate this process, it is essential to understand how legislative decision-making occurs in reality.

The U.S. Senate consists of 100 members, with two senators representing each state. Most senators are elected through statewide popular votes, but Maine and Nebraska use ranked-choice voting for their elections. In some cases, senators can be appointed by a state's governor if a seat becomes vacant. Legislative votes in the Senate can occur through roll call (where each senator's vote is recorded individually), unanimous consent (where legislation passes without objection), or voice vote (where senators respond collectively without individual records). The legislative process follows a structured path: a bill is introduced, undergoes committee review, then moves to the floor for debate in either the House or Senate. If passed, it must be approved by both chambers before reaching the president, who can either sign it into law or veto it, with Congress holding the power to override a veto with a two-thirds majority in both houses. 

<p align="center"><b>Summary: Introduced → Committee Review → Floor Debate (House or Senate) → Passed by Both Chambers → President's Signature (becomes law) or Vetoed (and potentially overridden)</b></p>

However, decision-making in the Senate is not purely procedural- it is shaped by party influence, bipartisan negotiations, lobbying from interest groups, backroom deals, and constituency pressure. Party-line voting in the U.S. Senate has become increasingly prevalent in recent years. In 2022, a record 83.1% of Senate votes were divided along party lines, surpassing the previous high of 79.2% set in 2021. (Kelly 2023) They may also form coalitions on shared interests. Lobbying organizations and corporations exert influence through campaign funding and policy research, while informal negotiations allow lawmakers to trade votes or attach amendments that benefit their states. Public opinion and presidential influence further shape legislative outcomes, making Senate decision-making a complex interplay of institutional rules, political strategy, and external pressures (challenges that make computational simulation particularly difficult).

## Data
For this study, an original dataset was created by webscraping, collecting variables across two dimensions:
1. Demographic and ideological characteristics of the 118th US senators.
2. Contextual variables for 45 floor votes, including how each senator voted.

The final dataset has 4692 rows (one vote per senator per floor vote) and 51 columns.

### Data Sources
**For senators:**
- *Congress.gov (“Congress.gov | Library of Congress” 2025):* This website serves as the official source for U.S. federal legislative information and was utilized to obtain basic biographical details on senators, including name, state, party affiliation, time served in the House, and time served in the Senate.
- *The Lugar Center (“Bipartisan Index” 2023):* The Lugar Center is a non-profit founded by former United States Senator Richard G. Lugar. It was used to obtain the Bipartisan Index, "an objective measure of how well members of opposite parties work with one another using bill sponsorship and co-sponsorship data." The index assigns scores based on how a member’s bipartisan bill sponsorship and co-sponsorship compare to a 20-year historical baseline, with positive scores indicating above-average bipartisanship and negative scores indicating below-average performance. Scores above 1.0 are considered outstanding, while those below -1.0 are very poor. Raw data is sourced from GovTrack, and certain exclusions apply: members serving less than 10 months, minority/majority leaders, and those sponsoring fewer than three qualifying bills (since the 113th Congress) are not scored to ensure fair comparisons.
- *Voteview (“Voteview | Congress View” 2025):* Voteview, maintained by UCLA's Department of Political Science and Social Science Computing, provides ideological position scores for legislators based on roll call voting records. These scores are computed using the DW-NOMINATE (Dynamic Weighted NOMINAl Three-step Estimation) method, which maps legislators onto a spatial scale where ideological similarity is reflected by proximity. The primary dimension, consistent throughout American history, represents a left-right (liberal-conservative) spectrum, while a second dimension captures intra-party differences on issues like civil rights and economic policy. Scores typically range from -1 (most liberal) to +1 (most conservative), allowing for quantitative analysis of congressional polarization and voting behavior.
- *CQ Press Congress Collection (“CQ Congress Collection” 2025):* A comprehensive research database containing biographical, political, and electoral data on every member of Congress since the 79th Congress (1945). This resource was utilized to obtain biographical details such as date of birth, education level, religion, race, and sex; electoral data including the percentage of votes received in the most recent election; and political data such as voting frequency, party alignment, instances of voting with or against party positions, and presidential support. Additionally, the database provides narrative biographies authored by CQ editorial staff.
- *The Cook Political Report (“The 2023 Cook Partisan Voting Index (Cook PVISM)” 2023):* Introduced in 1997, the Cook Partisan Voting Index (Cook PVI) measures how each state and district performs at the presidential level compared to the nation as a whole. The dataset was utilized to get the 2023 Cook Partisan voting score for the state of each senator. The 2023 PVI scores were calculated using 2016 and 2020 presidential election results. The Cook PVI scores indicate a state's partisan lean by comparing its average Democratic or Republican performance to the national average, expressed in terms such as D+5 (Democratic-leaning by 5 points) or R+3 (Republican-leaning by 3 points). A score of EVEN signifies a state that votes in line with the national average.

**For floor votes:**

A total of 45 floor votes were selected for simulation, including votes on motions such as "On Motion to Discharge Committee," "On Overriding the Veto," "On Passage of the Bill," and "On Cloture on the Motion to Proceed." These votes were randomly chosen from a curated list comprising CQ Key Votes, which are identified by CQ's editorial staff as the most significant floor votes in determining legislative outcomes, as well as key pieces of legislation highlighted on Wikipedia and the most-viewed bills on Congress.gov. The selection included bills originating in both the House and Senate, as well as joint resolutions, but was limited to votes that involved at least one roll call vote in the Senate.
- *Congress.gov (“Congress.gov | Library of Congress” 2025):* This website was utilized for obtaining contextual information about each bill, including its title, summary, number of co-sponsors, the party and name of the introducing senator, and its policy area. Bill summaries are authored by the Congressional Research Service ("CRS provides Congress with analysis that is authoritative, confidential, objective, and non-partisan.")
- *Senate.gov (“U.S. Senate: Roll Call Votes 118th Congress” 2024):* This website was utilized to obtain detailed records of senators' voting behavior during floor votes, including whether they voted "yea," "nay," were present, or did not vote.

### Target
The target variable being predicted is a senator's vote during a specific floor vote. This can take 4 values: Yea, Nay, Present or Not Voting.
- Yea: A vote in favor of a proposal.
- Nay: A vote against a proposal.
- Not Voting: A senator chooses not to vote (either absent or abstaining).
- Present: A senator is physically present but chooses not to vote either for or against the proposal.

### Features
The features fed into the LLM simulation were:
1. #### For Senators:
   - name – Senator's name
   - age – Senator's age
   - religion – Senator's religious affiliation (or Unaffiliated/Not Mentioned)
   - education – Senator's highest level of education
   - party – Political party affiliation
   - state – State represented by the senator
   - state_pvi – Cook Partisan Voting Index for the senator's state
   - years_house – Total years served in the U.S. House of Representatives
   - years_senate – Total years served in the U.S. Senate
   - last_election – Percentage of vote secured in senator's last election (or appointed)
   - party_loyalty – Measure of how often the senator votes in line with their party
   - party_unity – Measure of the senator's alignment with their party on key votes
   - presidential_support – Measure of how often the senator votes in support of the president’s agenda
   - voting_participation – Rate of participation in votes
   - dw_nominate – Ideological score (-1 to +1, liberal to conservative)
   - bipartisan_index – Measure of how frequently the senator works with the opposing party
   - backstory – Senator's biographical background

2. #### For Bills:
   - vote_date – Date of the floor vote
   - type_vote – Type of vote (e.g., Passage, Cloture)
   - measure_summary – Summary of the bill under consideration
   - sponsor – Name of the senator who introduced the bill
   - introduced_party – Political party affiliation of the sponsor
   - num_cosponsors – Number of senators who cosponsored the bill
   - previous_action – All prior legislative actions related to the bill

### Data Wrangling
**Handling Missing variables:**
- The Bipartisan Index was unavailable for the then Senate Majority (Charles E. Schumer) and Minority Leaders (Mitch McConnell). These values were left missing rather than imputed.
- Nebraska and Maine report different Cook Partisan Voting Index (PVI) scores for their respective congressional districts. To ensure consistency, the average PVI for the state was taken.
- Missing biographical and electoral data from the CQ Press Congress Collection (such as for Laphonza Butler) were manually verified and filled using news reports.
- Missing bill summaries (specifically for S.J.Res.51 and H.R.10545) were generated using an LLM based on bill text.

**Feature Engineering:**
To avoid multicollinearity in classification models and simplify input for LLM, following ratios were constructed: 
- Party Loyalty Ratio: Calculated as the proportion of votes cast in alignment with the senator’s party: (Votes with Party) / (Total Votes Cast)
- Party Defection Ratio: Calculated as the proportion of votes cast against the senator’s party: (Votes Against Party) / (Total Votes Cast)
- Key Vote Ratios: The same loyalty and defection ratios were calculated specifically for key votes to capture differences in party alignment on high-profile legislation. 

**Preprocessing for Classification Models:**
- The target variable (vote) was encoded as 1 for yea and 0 for nay. All other vote types were excluded.
- Session extraction: The legislative session (1 or 2) was determined based on the year recorded in the vote date.
- Religion categorization: Individual religious affiliations were grouped into broader categories for analytical consistency.
- Label Encoding was applied to: Party affiliation, education level, state partisan direction and bill sponsor's party.
- One-Hot Encoding was applied to: Religion, race and topic of the bill.

### Exploratory Data Analysis (EDA)
<p align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/eda/senators_eda1.png?raw=true" width="600"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/eda/senators_eda2.png?raw=true" width="600"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/eda/senators_eda3.png?raw=true" width="600"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/eda/bills_eda.png?raw=true" width="600"/>
</p>

### Variable Codebook
#### senators_data
| Variable Name       | Data Type  | Description | Source |
|---------------------|-----------|-------------|---------|
| first_name         | String    | Senator's first name | Congress.gov |
| last_name          | String    | Senator's last name | Congress.gov |
| TotalAll           | Integer   | Total votes cast | CQ Congress Collection |
| AllVoteW           | Integer   | Votes with party | CQ Congress Collection |
| AllVoteO           | Integer   | Votes against party | CQ Congress Collection |
| TotalKey           | Integer   | Total key votes cast | CQ Congress Collection |
| KeyVoteW           | Integer   | Key votes with party | CQ Congress Collection |
| KeyVoteO           | Integer   | Key votes against party | CQ Congress Collection |
| date_of_birth      | Date      | Date of birth | CQ Congress Collection |
| education          | String    | Highest degree attained | CQ Congress Collection |
| education_category | String    | Highest level of education attained (highschool, associates, undergraduate or postgraduate) | CQ Congress Collection |
| state              | String    | State represented | Congress.gov |
| state_pvi         | Float     | Cook Partisan Voting Index for senator's state. Negative means more conservative. Positive means more liberal.| Cook Political Report |
| party             | String    | Political party affiliation | Congress.gov |
| start             | Date      | Start date of current term | Congress.gov |
| end               | Date      | End date of current term | Congress.gov |
| served_house      | String    | Range of years served in the U.S. House (if applicable) | Congress.gov |
| years_house       | Integer   | Total years served in the House | Congress.gov |
| served_senate     | String    | Range of years served in the Senate | Congress.gov |
| years_senate      | Integer   | Total years served in the Senate | Congress.gov |
| dw_nominate       | Float     | Ideological score (-1 to +1, liberal to conservative) | Voteview |
| bipartisan_index  | Float     | Bipartisan Index score | The Lugar Center |

#### bills_data
| Variable Name       | Data Type  | Description | Source |
|---------------------|-----------|-------------|---------|
| vote_date         | Date      | Date of the vote | Senate.gov |
| measure_number    | String    | Bill or resolution number | Congress.gov |
| vote_result      | String    | Outcome of the vote (e.g., passed, rejected) | Senate.gov |
| previous_action   | String    | Summary of prior legislative actions | Congress.gov |
| required_majority | Integer   | Number of votes required for passage | Senate.gov |
| type_vote        | String    | Type of vote (e.g., Passage, Cloture) | Senate.gov |
| measure_title    | String    | Official title of the measure | Congress.gov |
| measure_summary  | String    | Summary of the measure authored by CRS | Congress.gov |
| bill_text        | String    | Full text of the bill | Congress.gov |
| yea             | Integer   | Number of "yea" votes | Senate.gov |
| nay             | Integer   | Number of "nay" votes | Senate.gov |
| not_voting      | Integer   | Number of senators not voting | Senate.gov |
| sponsor         | String    | Name of the bill's primary sponsor | Congress.gov |
| introduced_party | String    | Party affiliation of the sponsor | Congress.gov |
| num_cosponsors  | Integer   | Number of cosponsors | Congress.gov |
| topic           | String    | Primary policy area of the bill | Congress.gov |

#### For each vote
| Variable Name       | Data Type  | Description | Source |
|---------------------|-----------|-------------|---------|
| first_name         | String    | Senator's first name | Senate.gov |
| last_name          | String    | Senator's last name | Senate.gov |
| party             | String    | Political party affiliation | Senate.gov |
| state              | String    | State represented | Senate.gov |
| vote              | String    | Vote casted (yea, nay, present or not voting)  | Senate.gov |

## Research Design
### Overview
![Thesis Presentation](https://github.com/user-attachments/assets/1a9a42a7-ba5b-4735-8e90-ecf3d5a4048f)
The simulation reconstructs 45 roll call votes that took place in the 118th U.S. Senate between January 3, 2023, and January 3, 2025. It models individual senators as agents, each instantiated with biographical attributes, political affiliations and backstories to approximate real-world behavior.The study follows a structured multi-stage approach to simulate Senate voting dynamics:
- **Initial Vote:** Senators cast an initial vote based solely on basic bill characteristics (bill summary, date of the vote, number of co-sponsors, name and party affiliation of the bill’s primary sponsor, all legislative actions on the bill up to that point). Senators are only allowed to vote “yea” or “nay” (abstentions and "present" votes are excluded).
- **Debate Phase:** Senators engage in two rounds of debate in a round-robin format, ensuring all participants have an opportunity to contribute and respond to the debate at least once.
- **Final Vote:** Following the debate, senators cast a final vote based on all previous bill characteristics and a LLM generated conversation summary. Again, only “yea” or “nay” votes are permitted.
- **Performance Evaluation:** Votes cast before and after the debate are compared with actual voting records. Any LLM-generated response that deviates from "yea" or "nay" votes is categorized as "maybe." Accuracy is assessed by measuring how closely the simulated votes align with actual voting records. The final outcome of the simulation (passed/rejected) is calculated based on Senate voting thresholds (1/2, 2/3, or 3/5 majority) and compared to the actual legislative result.

### Selection of simulation context
The 118th U.S. Senate was chosen as the simulation environment due to its recentness, comprehensive data availability, and diverse political landscape. According to a 2023 analysis by the Pew Research Center, the "118th Congress is the most racially and ethnically diverse in U.S. history" (Geiger 2023), making it an ideal context for evaluating contemporary legislative dynamics.

At the start of the 118th Congress, the Senate consisted of: 52 Democrats (including the Vice President), 4 Independents and 49 Republicans. The study accounts for six membership changes that occurred during the session, ensuring accuracy in the composition of the simulated Senate. Ben Sasse resigned January 8, 2023 and was replaced by Pete Rickettes starting January 23, 2023. Joe Manchin III changed his party affiliation to independent. Dianne Feinstein died September 29, 2023 and Laphonza Butler was appointed October 3, 2023 to replace her. Bob Menendez resigned August 20, 2024 and was replaced by George Helmy starting September 9, 2024. Finally, Butler and Helmy resigned from their appointments December 8, 2024 to allow successors Andy Kim and Adam Schiff to take office early.

The first session took place between January 3, 2023 – January 3, 2024 and the second session took place between January 3, 2024 – January 3, 2025. According to Congress.gov, in the Senate there were 352 roll call votes in the first session and 339 roll call votes in the second session. Democrats remained the majority party throughout when caucusing with Independents.

| Date | Democrats | Independents | Republicans | Total Seats | Vacant |
|-------|-----------|--------------|------------|-------------|--------|
| End of 117th Congress | 48 | 2 | 50 | 100 | 0 |
| January 3, 2023 | 48 | 3 | 49 | 100 | 0 |
| January 8, 2023 | 48 | 3 | 48 | 99 | 1 |
| January 23, 2023 | 49 | 3 | 48 | 100 | 0 |
| September 29, 2023 | 47 | 3 | 49 | 99 | 1 |
| October 3, 2023 | 48 | 3 | 49 | 100 | 0 |
| May 31, 2024 | 47 | 4 | 49 | 100 | 0 |
| August 20, 2024 | 46 | 4 | 49 | 99 | 1 |
| September 9, 2024 | 47 | 4 | 49 | 100 | 0 |

The simulation focuses exclusively on roll call votes, particularly those cast on overriding a presidential veto, motions to discharge a bill from committee, motions to invoke cloture (to end debate), and votes on final passage of a bill. Roll call votes were selected due to their clear documentation and formal recording of each senator’s position. These votes occur when at least one-fifth of a quorum requests them, ensuring transparency. In most cases, a simple majority (1/2) is required for passage, though in the event of a tie, the vice president may cast the deciding vote. Certain measures require a higher threshold: a two-thirds majority (2/3) is needed to override a presidential veto, propose constitutional amendments, convict an impeached official, ratify treaties, or expel a senator. Additionally, a three-fifths majority (3/5) is required to invoke cloture and end debate on legislation. (“U.S. Senate: About Voting” 2024)

### Selection of LLM model and memory-handling
Two different LLMs were used to facilitate the simulation. **Meta’s Llama-3.3-70B-Versatile** was used for the debate simulation, as it offers a 128K context window and a maximum completion length of 32,768 tokens. For generating the debate summary, **Meta’s Llama-3.1-8B-Instant** was selected, with a 128K context window and a maximum completion length of 8,192 tokens. (Meta-Llama 2025) These models were chosen primarily for their large context windows, which allow for long-form discussions with minimal information loss and easy memory handling.

Additionally, the models’ knowledge cutoff in December 2023 ensured that votes from 2024 were unknown to them, allowing the study to test predictive reasoning rather than model memorization. The models were accessed via the Groq API, which provides significantly faster inference speeds (due to its hardware specifically designed for single-batch, high-throughput processing) compared to traditional cloud-based deployments. Faster inference was particularly crucial given the large number of agents involved and the need for maintaining continuity across multiple rounds of debate and voting.

Due to resource constraints, a context window limit of 15,000 tokens was imposed during debates. Given that each senator’s response averaged approximately 300 tokens, this meant that once the total conversation history exceeded 15,000 tokens, the oldest messages were deleted. As a result, senators who spoke later in the debate only had access to the most recent portion of the discussion, typically around 25 percent of the total debate content.

### Selection of multi-agent framework
The simulation was implemented using the LlamaSim multi-agent framework (Wu 2025), a system designed specifically for modeling large-scale human interactions. The framework simulates how a given target population (such as voters or students) responds to specific questions/events. LlamaSim was selected because it is optimized for structured simulations of real-world social contexts, unlike alternatives such as CrewAI and LangChain, which experience significant latency issues when handling large-scale interactions. Additionally, LlamaSim integrates seamlessly with fast inference providers such as Groq and Cerebras, reducing the computational bottlenecks typically associated with multi-agent modeling.

### Determining the order in which agents will interact
Within LlamaSim, agents can interact either in a random order or in a structured round-robin format. The round-robin approach was selected because it ensures that every senator has an opportunity to participate in the debate. This method also closely resembles the structured speaking order used in real-world Senate deliberations, preventing a small subset of senators from dominating the conversation while allowing for a more balanced exchange of ideas.

### Controlling model parameters
To ensure the reliability of the simulation, several key model parameters were controlled. The system prompt was designed to respect the models’ knowledge cutoff of December 2023, ensuring that LLM-generated responses were based on reasoning rather than memorization. This cutoff was particularly useful, as it prevented the models from having prior knowledge of the actual votes cast in 2024, allowing for an unbiased evaluation of predictive reasoning.

Temperature settings were also adjusted to optimize model performance. For debate simulation, a temperature of 0.4 was used to allow for a moderate degree of randomness, fostering a diversity of arguments and reasoning styles. For debate summary generation, a lower temperature of 0.2 was set to make the output more deterministic and fact-based. Additionally, a fixed random seed was used for the reproducibility of results across multiple simulation runs (although due to the inherent output variability, setting a random seed in LLMs does not always guarantee deterministic outputs).

## Methodology
### Correlations and Classifications
To explore the predictive capability of various features on voting behavior, correlations between different variables and the target outcome (vote) were analyzed. The goal was to identify the strongest predictors of legislative decision-making. A correlation matrix was constructed to examine the relationships between key numerical variables, and pairwise scatterplots were used to visualize patterns in the data.

To assess classification accuracy, 12 different statistical models on the dataset were tested. Each model was trained and evaluated using cross-validation as well as a train-test split to determine its predictive performance in classifying voting outcomes. The models were evaluated by calculating the F1 score and ROC curve for each. Feature importance for all models was also calculated to analyze weights. The hyperparameters for the best performing models were tuned using GridSeachCV. The models included Naïve Bayes, k-Nearest Neighbors, Logistic Regression, Linear Support Vector Classification, Support Vector Classification, Decision Tree Classifier, Extra Tree Classifier, Extra Trees Classifier, AdaBoost Classifier, Random Forest Classifier, Perceptron and Multi-layer Perceptron (MLP) Classifier. These models allowed us to compare how forecasting through LLMs compares to predictions made through traditional classification methods.

Formula:
<p align = "middle">
Vote = β₀ + β₁ * Required Majority + β₂ * Introduced Party + β₃ * Number of Cosponsors + β₄ * Education Category + β₅ * Party Unity Support + β₆ * Voting Participation + β₇ * Percentage Scored in Last Election + β₈ * Age + β₉ * Sex + β₁₀ * State Direction + β₁₁ * State PVI + β₁₂ * Party + β₁₃ * Years in House + β₁₄ * Years in Senate + β₁₅ * DW-Nominate + β₁₆ * Bipartisan Index + β₁₇ * Month + β₁₈ * Is Election Season + β₁₉ * Session + β₂₀ * Passage Vote Or Not + β₂₁ * Party Loyalty + β₂₂ * Party Defection + β₂₃ * Key Vote Defection + β₂₄ * Religion + β₂₅ * Race + β₂₆ * Topic of the Bill + ε </p>

Where:
- **Vote**: Predicted vote of the senator (1 or 0).
- **β₀**: Intercept of the model.
- **β₁, β₂, ..., β₂₆**: Coefficients for each independent variable.
- **ε**: Error term representing unobserved factors affecting the vote.

### Basic Simulation (Without Prompt Engineering or Parameter Controls)
A baseline simulation was conducted in which the system prompt was kept minimal: "You are US senator {full_name}." Senators were only provided with a bill summary and asked to vote. No additional contextual information, deliberation, or parameter tuning was included in this stage. A total of 30 floor votes were simulated in this manner, focusing solely on passage votes.

### Simulation with Prompt Engineering and Parameter Controls
To improve the accuracy and realism of the simulation, additional prompt engineering was introduced in the next phase. This involved modifying the system prompt to incorporate factors such as political ideology and party alignment in the development of each agent. More context about the bill such as the date of the vote and previous action was also given. Specific parameters, such as temperature were controlled to have moderate variability in output.

### Simulation with prompt engineering, with controlling parameters and after 2 rounds of debate
The final simulation incorporated both prompt engineering and structured deliberation. Each senator engaged in two rounds of debate before casting their final vote. The purpose of this was to observe how deliberation influenced decision-making and whether the debate led to shifts in voting behavior. Additionally, the impact of debate on model accuracy was assessed by comparing predicted votes before and after discussion.

### Determining Variable Weights via Regression Analysis
To gain insight into the factors influencing prediction accuracy across various simulations, an attempt was made using multiregression models. The dependent variable in these models was the individual vote prediction accuracy for each bill. Independent variables included various bill characteristics, such as the number of cosponsors, the required majority threshold, and the status of the election season. This analysis facilitated a basic understanding of the correlations between bill features and the outputs generated by the LLM. However, there is a significant limitation here: since we do not have access to all the features that the LLM used in making its predictions, in order to comprehensively understand the reasoning behind the LLM's outputs, it is essential to fine-tune the model or request explanations for its predictions.

Formula:

<p align="middle">
<i></i>Simulation Accuracy = β₀ + β₁ * Year + β₂ * Final Vote Result + β₃ * Previous Action Length + β₄ * Required Majority + β₅ * Measure Summary Length + β₆ * Number of Senators Not Voting + β₇ * Number of Cosponsors + β₈ * Margin Between Yeas and Nays + β₉ * Introduced Party + β₁₀ * Topic + β₁₁ * Type of Vote + ε</i></p>

Where:
- **β₀** is the intercept of the regression model.
- **β₁, β₂, ..., β₁₁** are the coefficients for each independent variable.
- **ε** is the error term.

## An AI Senator
This is what an LLM-powered senator looks like:
![AI_Agent_Slide-1](https://github.com/user-attachments/assets/7cc77339-7f24-448d-83b2-25c937e1ff9a)

## A Sample Simulation
<p float="left" align="middle">
<img src="https://github.com/user-attachments/assets/09cefb2e-1701-4434-8a83-694719d98f03" width="600"/>
<img src="https://github.com/user-attachments/assets/03962bfa-d0c1-4c08-ad24-a2223d0215a4" width="600"/>
<img src="https://github.com/user-attachments/assets/15400582-18d9-4a41-af65-104856f574ff" width="600"/>
</p>

## Results
### Correlations
<p float="left" align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/eda/correlations_feature_target.png?raw=true" width="600"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/eda/scatterplots.png?raw=true" width="600"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/eda/scatterplots2.png?raw=true" width="600"/>
</p>

The correlation matrix and scatterplot indicate a strong negative correlation of -0.78 between DW-Nominate scores and State PVI scores. This suggests that as a state becomes more Republican-leaning, politicians from that state tend to exhibit more conservative DW-Nominate scores and vice-versa for Democrat-leaning states, which makes sense.


Additionally, there is a negative correlation of -0.39 between party loyalty and DW-Nominate scores, suggesting that more liberal senators may cast votes that align more closely with their party. The scatterplot also shows that as the bipartisan index increases, party loyalty decreases, indicating that senators who are more willing to cross party lines and engage in bipartisan efforts tend to exhibit lower levels of party loyalty.

For the target variable (vote), DW-Nominate and State PVI scores demonstrated the highest correlations, with values of 0.35 and -0.33, respectively. This supports the notion that voting along party lines is a significant factor.

### Classification Models
#### ROC Curves:
<p float="left" align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/Bernoulli.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/KNeighborsClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/LogisticRegression.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/DecisionTreeClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/ExtraTreeClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/ExtraTreesClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/RandomForestClassifer.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/LinearSVC.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/SVC.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/Perceptron.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/MLPClassifer.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/roc_curves/AdaBoostClassifier.png?raw=true" width="32%"/>
</p>

#### Feature Importance:
<p float="left" align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/Bernoulli.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/KNeighborsClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/LogisticRegression.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/DecisionTreeClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/ExtraTreeClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/ExtraTreesClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/RandomForestClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/LinearSVC.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/SVC.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/Perceptron.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/MLPClassifier.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/models/feature_importance/AdaBoostClassifier.png?raw=true" width="32%"/>
</p>

| Model                      | Cross-validation Accuracy | Test Accuracy | F1 Score | ROC AUC |
|----------------------------|--------------------------|---------------|----------|---------|
| BernoulliNB                | 0.72 ± 0.03              | 0.71          | 0.71     | 0.76    |
| KNeighborsClassifier        | 0.88 ± 0.01              | 0.88          | 0.88     | 0.94    |
| LogisticRegression          | 0.80 ± 0.03              | 0.81          | 0.806     | 0.81    |
| LinearSVC                  | 0.80 ± 0.03              | 0.80          | 0.80     | 0.81    |
| SVC                        | 0.89 ± 0.02              | 0.91          | 0.91     | 0.95    |
| DecisionTreeClassifier      | 0.91 ± 0.02              | 0.92          | 0.92     | 0.92    |
| ExtraTreeClassifier         | 0.87 ± 0.02              | 0.87          | 0.87     | 0.86    |
| ExtraTreesClassifier        | 0.93 ± 0.01              | 0.92          | 0.93     | 0.98   |
| AdaBoostClassifier          | 0.77 ± 0.04              | 0.80          | 0.80     | 0.80    |
| RandomForestClassifier      | 0.93 ± 0.01              | 0.92          | 0.93     | 0.98    |
| Perceptron                 | 0.70 ± 0.04              | 0.72          | 0.712     | 0.74    |
| MLPClassifier               | 0.92 ± 0.02              | 0.92          | 0.92     | 0.97      |

Overall, the ExtraTreesClassifier and RandomForestClassifier exhibited the highest performance metrics. Fine-tuning these two models based on their optimal parameters resulted in a test accuracy of 0.93 for both. The MLPClassifier also performed well, achieving a test accuracy of 0.92.

Analysis of feature importance revealed that, for all these classifiers, several common factors held significant predictive power: the party affiliation of the senator, the party of the bill sponsor, the DW-Nominate score, State PVI, the topic of the bill, the required majority and the legislative session.

These high accuracy rates align with previous literature, which has demonstrated the effectiveness of ensemble models in modeling legislative behavior.

### Basic Simulation
In the initial vote simulation without any adjustments, the average model accuracy for individual votes was 58.11%. The average accuracy for predicting bill outcomes- whether a bill was passed or rejected- was significantly lower at 43.33%. The model demonstrated better performance on high-margin votes, characterized by strong "yea" or "nay" majorities, as well as on key legislative decisions. This improved accuracy can be attributed to the tendency of the LLM to predict uniform votes across the board.

Overall, these results indicate that the baseline performance of the model was only slightly better than that of a random binary model, which would yield an accuracy rate of around 50%. 

### Advanced Simulation
<p float="left" align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_1.png?raw=true" width="45%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_2.png?raw=true" width="45%"/>
</p>
Introducing structured prompts and parameter tuning led to a significant improvement in model accuracy. Before the debate, the average accuracy for individual votes reached 80.49%, but after deliberation, it declined to 66.38%, highlighting the impact of discussion on voting decisions.


<p align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_7.png?raw=true" width="50%"/>
</p>
The accuracy of bill predictions varied widely, ranging from 98% for the most accurately predicted bills to just 25% for the least accurate ones. Before debate, 23 out of 45 bills had an accuracy exceeding 90%. Interestingly, bills with close-margin votes (less than a five-vote difference between "yea" and "nay") tended to have the highest prediction accuracy, whereas those with large-margin votes (near-unanimous decisions) had the lowest accuracy. This pattern aligns with the findings of Halawi et al. (2024), whose forecasting system similarly struggled with over-hedging in high-certainty cases.


<p align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_5.png?raw=true" width="50%"/>
</p>
Overall, this advanced simulation demonstrated higher accuracy in predicting individual votes compared to the basic simulation.


<p float="left" align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_3.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_6.png?raw=true" width="32%"/>
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_8.png?raw=true" width="32%"/>
</p>
Debate influenced voting outcomes in several key ways:
- Accuracy decreased by 14.11% post-debate, suggesting that agents reconsidered and changed their decisions based on discussion.
- The percentage of simulated votes matching actual outcomes dropped slightly from 57.78% to 55.56% (the model did not do as well in predicting overall outcomes of the vote (passed or rejected) when compared to individual votes. This was largely due to the nature of close nature-margin votes- while the model could accurately predict most individual votes, small deviations led to misclassification of the final outcome.).
- The overall bill passage rate declined from 35.56% to 28.89% after debate.
- Post debate, there was an average decrease in 'yea' votes by 13.75% and a corresponding increase in 'nay' votes by 13.73%.


<p align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_4.png?raw=true" width="50%"/>
</p>
The most frequently used phrases in the debate were analyzed, with words such as "believe," "colleagues," "rule," "bill," "resolution," and "must" emerging as the most common after removing stopwords. Senators often repeated similar phrases and wording, which could be influenced by factors such as temperature settings in the model or access to previous senator comments. Additionally, some garbage outputs were observed during the debate, including unknown symbols or text in different languages, raising concerns about the quality of the output. Further NLP analysis of the generated discourse is needed to better understand these anomalies and the output.


<p align="middle">
<img src="https://github.com/Kulsoom-ri/US_senate_simulation/blob/main/results/advanced_LLM_simulation/Figure_10.png?raw=true" width="50%"/>
</p>
Notably, the model demonstrated relatively stable performance across legislative sessions from 2023 to 2024, suggesting that its reasoning process was not significantly influenced by the potential inclusion of this data in its training set, which had a cutoff date of December 2023. This indicates that the model was actively engaging with the debate and the data it was fed rather than merely recalling memorized outcomes.


### Regression Analysis

#### Before Debate:
| Metric                     | Value        |
|----------------------------|--------------|
| **Dependent Variable**     | before_debate |
| **R-squared**              | 0.929        |
| **Adj. R-squared**         | 0.734        |
| **F-statistic**            | 4.769        |
| **Prob (F-statistic)**     | 0.00451      |
| **No. Observations**       | 42           |
| **Log-Likelihood**         | -128.37      |
| **AIC**                    | 318.7        |
| **BIC**                    | 372.6        |

**Five Most Statistically Significant Features:**
*Variable, Coefficient, P-Value*
- final_vote_result, -51.3590, 0.000
- required_majority, -4.5758, 0.000
- topic_Civil Rights and Liberties, Minority Issues, 80.2503, 0.000
- topic_Environmental Protection, 58.3085, 0.000
- topic_Energy, 56.1799, 0.000
  

#### After Debate:
| Metric                     | Value        |
|----------------------------|--------------|
| **Dependent Variable**     | after_debate |
| **R-squared**              | 0.925        |
| **Adj. R-squared**         | 0.721        |
| **F-statistic**            | 4.527        |
| **Prob (F-statistic)**     | 0.00563      |
| **No. Observations**       | 42           |
| **Log-Likelihood**         | -137.39      |
| **AIC**                    | 336.8        |
| **BIC**                    | 390.6        |

**Five Most Statistically Significant Features:**
*Variable, Coefficient, P-Value*
- final_vote_result, -44.9427, 0.003
- topic_Civil Rights and Liberties, Minority Issues, 63.1719, 0.009
- topic_Environmental Protection, 47.7995, 0.005
- topic_Labour and Employment, 59.7548, 0.002
- topic_Animals, 67.8600, 0.001


#### Basic simulation:
| Metric                     | Value         |
|----------------------------|---------------|
| **Dependent Variable**     | basic_simulation |
| **R-squared**              | 0.923         |
| **Adj. R-squared**         | 0.464         |
| **F-statistic**            | 2.009         |
| **Prob (F-statistic)**     | 0.263         |
| **No. Observations**       | 29            |
| **Log-Likelihood**         | 42.453        |
| **AIC**                    | -34.91       |
| **BIC**                    | -0.7229       |

**Five Most Statistically Significant Features:**
*Variable, Coefficient, P-Value*
- introduced_party_republican, 0.5881, 0.096
- topic_Government Operations and Politics, -1.1206, 0.030
- topic_International Affairs, 0.5131, 0.071
- topic_Emergency Management, -0.8183, 0.057
- topic_Animals, 0.1297, 0.648

Model Performance:
- Before debate: The before-debate regression demonstrated a high R-squared value of 0.929, indicating that it explains approximately 92.9% of the variance in simulation accuracy before the debate. The adjusted R-squared of 0.734 shows robust explanatory power, suggesting the model is well-specified. The F-statistic of 4.769, with a p-value of 0.00451, indicates that the independent variables significantly predict the dependent variable. 
- After debate: The R-squared value of 0.925 indicates that the model explains approximately 92.5% of the variance in simulation accuracy after the debate. The adjusted R-squared of 0.721 also indicates that the model maintains its explanatory power even when accounting for the number of predictors used. The F-statistic of 4.527 with a significant p-value (0.00563) confirms that the model is statistically significant, meaning that the independent variables collectively have a meaningful relationship with the dependent variable.
- Basic simulation: The basic simulation regression showed fewer statistically significant factors. It has a R squared of 0.923, and a very low Adjusted R squared of 0.464, meaning that the model is very overfit. Also the P(F-statistic) is 0.263 which is much greater than 0.05, and means that the model as a whole is not statistically significant.

Key Predictors:
- The final vote outcome (passed or rejected) was statistically significant in the before debate and after debate simulations. This would mean that simulation accuracy was lower for bills that passed and the simulation accuracy was higher for bills that were rejected.
- The required majority (1/2, 3/5 or 2/3) was also statistically significant in predicting the before debate simulation accuracy.
- Several topic categories also show statistically significant coefficients for all simulation accuracies.

However, the presence of multicollinearity, as noted in the regression outputs, remains a critical concern. The model's overall fit, as indicated by the adjusted R-squared, suggests that other factors not captured in the model are influencing simulation accuracy. LLMs have billions of parameters, making direct analysis extremely difficult. A viable approach could be to fine-tune the LLM being used for simulation that would allow us to get some insight into how specific data is influencing its output. Chain-of-Thought Prompting (prompting the LLM to explicitly show its step-by-step reasoning process before providing a final answer) or self-explanation prompts could be other reasonable strategies to get some insight into explanability.

## Summary of findings
- For the target variable (vote), DW-Nominate and State PVI scores had highest correlations indicating that party-line voting is prevalent.
- Many classifier models performed exceptionally well in predicting the votes of senators, achieving a high test accuracy of 93%. This aligns with previous literature. The most important features included the party affiliation of the senator, the party of the bill sponsor, the DW-Nominate score, State PVI, the topic of the bill, the required majority, and the legislative session.
- Without tuning and prompt engineering, the LLM's accuracy was 58.11% for individual votes and 43.33% for overall bill outcomes, performing no better than a random model.
- Introducing parameter tuning and prompt engineering improved simulation accuracy. Pre-debate accuracy was 80.49%, dropping to 66.38% post-debate. Fewer bills were passed after the debate. Simulation accuracy was similar for both legislative sessions despite the LLM having a knowledge cutoff date in December 2023. The LLM seemed to performed better on close-margin votes, and senators often used similar phrases, warranting further NLP analysis.
- Statistically significant models for simulation accuracy before and after debates were developed, but significant limitations exist in using this approach for understanding the underlying mechanisms, necessitating further research.

## Discussion
### Significance of findings for Political Science:
**Development of Synthetic Data Simulations:** The research investigates the feasibility of using Large Language Models (LLMs) to create synthetic data simulations that resemble real-world outcomes. By leveraging the capabilities of LLMs, political scientists can generate vast datasets that mimic the behaviors and interactions of social groups. These sandbox simulations could provide an innovative approach to studying complex political and social dynamics without the ethical and practical constraints associated with real-world experiments.

**Accurate Replication of Human Decision-Making:** The study reveals that LLMs can potentially replicate human-made decisions with reasonable accuracy, achieving an overall prediction accuracy of 80.93% in simulating legislative voting behavior. By demonstrating the model's capability to produce results that align with actual outcomes, the study provides a potential foundation for integrating LLMs into the study of political behavior, thereby enhancing the predictive power of political science research.

**Testing New Methodologies for Simulating Legislative Processes:** The research tests a novel methodology for simulating legislative processes through the use of a multi-agent framework comprising 100 LLM agents. This methodology allows for the exploration of dynamic interactions among legislators, simulating real-time debates and decision-making processes that resemble those occurring in actual legislative bodies. The ability to simulate these processes in a controlled environment would be extremely useful in conducting "what-if" analyses, enabling political scientists to investigate hypothetical scenarios.

### Significance of findings for Computer Science
**Versatility of LLMs in Decision-Making Scenarios:** This study adds to the expanding research on the use of LLMs in applications beyond conventional language processing tasks. The findings demonstrate the potential of employing autonomous agents for complex decision-making scenarios, showcasing how LLMs can potentially simulate and analyze human-like decision-making processes.

**Importance of Model Fine-Tuning and Prompt Engineering:** The research emphasizes the vital role of model fine-tuning and prompt engineering in optimizing the performance of AI systems for specific tasks. By refining model parameters and carefully designing input prompts, researchers can substantially enhance the accuracy and reliability of LLM outputs, making them more suitable for practical applications.

**Addressing Explainability and Reliability Concerns:** The results underscore the significant challenges related to the explainability and reliability of outputs generated by LLMs. As these AI systems become more prevalent in critical decision-making areas, including politics and governance, it is crucial to establish frameworks that clarify the reasoning behind LLM-generated results. As LLM technology advances, it is essential to tackle the "black box" issue to ensure that users can trust and comprehend the decisions made by these models.

### Limitations
This study has several significant limitations that must be acknowledged:
1. Limited Scope of Simulation: Only one U.S. Senate (the 118th) was simulated, which restricts the generalizability of the results. A total of 45 floor votes were simulated out of the 691 floor votes that occurred during the 118th Senate, indicating a limited dataset that may not fully represent the dynamics of Senate voting behavior.
2. Context Window Constraints: The context window for the debate simulation was capped at 15,000 tokens due to resource constraints. This limitation may have influenced the post-debate simulation outcomes as not all senators were getting full access to the shared context.
3. Single Language Model Utilization: Only one large language model (LLM) was employed for the debate simulation. Research has shown that different LLMs can have varying political preferences and moral biases depending on their training dataset (Rozado 2024). Future studies should explore the generalizability of results by testing different LLMs, as varying architectures and training datasets may yield different results.
4. Single Seed and Temperature: The simulations were conducted with a fixed seed and temperature. LLMs can give varying answers at different seeds depending on how "confused" they are (although good prompts should yield similar results across seeds). Future work should involve testing the simulations across different seeds and temperature settings to assess the robustness and reliability of the findings.
5. Challenges in Explainability: Explainability remains a significant challenge in this study. Understanding how the LLM arrived at its predictions and outputs is crucial for evaluating the model's validity and reliability.
6. Lack of Fine-Tuning: The study did not involve fine-tuning the LLM based on the specific context of U.S. Senate votes. Fine-tuning could enhance the model's predictive capabilities and improve simulation performance.
7. Simplified Legislative Process Simulation: The simulation oversimplifies the legislative process by only simulating two rounds of debate and a single floor vote. The complexity of how bills progress through Congress is substantial, as many bills do not advance past the introduction stage and involve numerous committees and negotiations before reaching the floor for a vote. It is also extremely difficult to capture the nuances of backroom deals and strategic alliances in the simulation. Senators often engage in informal discussions and negotiations that influence the final outcome of a bill. 
8. Garbage Outputs in Debate Simulation: Some garbage outputs were generated during the debate simulation and were not filtered out. This could have impacted how agents reacted to the debate or how the conversation summary was generated.

### Future Work
Future work in this area can focus on several key improvements and analyses. First, a comprehensive natural language processing (NLP) analysis of the text generated during simulations is necessary to assess its coherence and relevance, as well as to investigate intriguing insights, such as when senators are most likely to switch votes, which arguments they agreed with or disagreed with, and why there were more "nay" votes after the debate. Second, evaluating the believability of the debates is crucial. Developing metrics to assess the believability and realism of the simulated discussions will enhance understanding of the model's effectiveness in capturing authentic legislative dialogue.

On the technical side, refining the model's memory handling capabilities represents another important step. Improvements in this area will help maintain contextual awareness during extended discussions, thereby enhancing the quality of interactions in simulations and ensuring that the generated debates remain coherent and contextually relevant.

Additionally, future work should focus on creating a fine-tuned large language model (LLM) that utilizes the data collected from this study as a training set. Research indicates that foundational language models demonstrate robust pattern-recognition abilities, which can lead to significant performance improvements on specific tasks with relatively minimal additional training (Google for Developers, 2025). It is anticipated that a fine-tuned model will substantially enhance the existing simulation.

Finally, the explainability of the model- specifically, how the LLM arrives at its conclusions during the simulation- needs to be thoroughly analyzed. Benchmarks should be constructed to evaluate the model's performance in comparison to human experts in forecasting individual votes of senators and the overall outcome of a bill. This evaluation would enhance our understanding of the model's decision-making process and contribute to the applicability of such methodologies for forecasting or to synthetically simulate different population groups.

## File Description
This github repository contains all the code and data utilized to generate results for this study.

| Directory/File                  | Description                                                                                          |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| **data/**                       |                                                                                                      |
| roll_call_votes.xlsx            | Contains data on Senate roll call votes (1/3/2023 - 1/3/2025), including description and vote tally.            |
| senators_data.xlsx              | The main dataset. Contains information on senators and floor votes simulated in this study               |
| **eda/**                        | Contains script for exploratory data analysis (EDA) of senators_data.xlsx and the generated visualizations. |
| **results/**                    |                                                                                                      |
| accuracy_regression.txt         | Text file containing the regression analyses with simulated accuracies as target variables.                                  |
| accuracy_results.xlsx           | Excel file containing the accuracy of simulation in different phases for each bill.                              |
| advanced_LLM_simulation/        | Contains result visualizations from the advanced simulations.          |
| basic_LLM_simulation/           | Contains output for basic simulation.                                         |
| models/           | Contains output for basic_inference.py. Also visualizations of ROC curves and feature importance for each of the 12 statistical models.                                         |
| **simulation output/**          | Contains outputs generated from the simulations, including debate and tabulated accuracies.         |
| **webscrapers/**                |                                                                                                      |
| bills_reformatting.py          | Script to reformat and clean up scraped bill data for analysis.                                     |
| cqpress_scrape.py              | Script to scrape legislative data from CQ Press.                                                    |
| senators_scrape.py              | Script to scrape senator-related data from congress.gov.                                            |
| **README.md**                   | Documentation file.   |
| agent.py                        | Final code to instantiate and simulate a single agent.                      |
| basic_inference.py            | Script to run 12 classification models on the dataset.             |
| network.py                      | Final code to set up the network interactions.                   |
| prompts.txt                     | File containing prompts used for instantiating agents and generating debate.        |
| results.py                      | Script to generate result visualizations and compile outputs from the simulations.                                 |
| run.py                          | Final script for executing the complete simulation process.                                         |
| run_basic.py                    | Script for executing basic simulation.                                                |

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
- Luo, Xiaoliang, Akilles Rechardt, Guangzhi Sun, Kevin K Nejad, Felipe Yáñez, Bati Yilmaz, Kangjoo Lee, et al. 2024. “Large Language Models Surpass Human Experts in Predicting Neuroscience Results.” Nature Human Behaviour, November. https://doi.org/10.1038/s41562-024-02046-9.
- Wolff, Malcolm L, Shenghao Yang, Kari Torkkola, and Michael W Mahoney. 2025. “Using Pre-Trained LLMs for Multivariate Time Series Forecasting.” ArXiv.org. 2025. https://arxiv.org/abs/2501.06386.
- Kosinski, Michal. 2024. “Evaluating Large Language Models in Theory of Mind Tasks.” Proceedings of the National Academy of Sciences 121 (45). https://doi.org/10.1073/pnas.2405460121.
- West, Peter, Ximing Lu, Nouha Dziri, Faeze Brahman, Linjie Li, Jena D Hwang, Liwei Jiang, et al. 2023. “The Generative AI Paradox: ‘What It Can Create, It May Not Understand.’” ArXiv.org. 2023. https://arxiv.org/abs/2311.00059.
- Shumailov, Ilia, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. 2024. “AI Models Collapse When Trained on Recursively Generated Data.” Nature 631 (8022): 755–59. https://doi.org/10.1038/s41586-024-07566-y.
- Banerjee, Sourav, Ayushi Agarwal, and Saloni Singla. 2024. “LLMs Will Always Hallucinate, and We Need to Live with This.” ArXiv.org. 2024. https://arxiv.org/abs/2409.05746.
- Kelly, Ryan. 2023. “2022 Vote Studies: Division Hit New High in Senate, Fell in House - Roll Call.” Roll Call. March 24, 2023. https://rollcall.com/2023/03/24/2022-vote-studies-division-hit-new-high-in-senate-fell-in-house/?utm_source=chatgpt.com.
- “Congress.gov | Library of Congress.” 2025. Congress.gov. 2025. https://www.congress.gov/.
- “Bipartisan Index” 2023. Thelugarcenter.org. 2023. https://www.thelugarcenter.org/ourwork-Bipartisan-Index.html.
- “Voteview | Congress View.” 2025. Voteview.com. 2025. https://voteview.com/congress/senate/-1/text.
- “CQ Congress Collection.” 2025. Duke.edu. 2025. https://library-cqpress-com.proxy.lib.duke.edu/congress/.
- “U.S. Senate: Roll Call Votes 118th Congress” 2024. Senate.gov. February 8, 2024. https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_118_1.htm.
- Geiger, Abigail. 2023. “The Changing Face of Congress in 8 Charts.” Pew Research Center. February 7, 2023. https://www.pewresearch.org/short-reads/2023/02/07/the-changing-face-of-congress/.
- “U.S. Senate: About Voting.” 2024. Senate.gov. December 27, 2024. https://www.senate.gov/about/powers-procedures/voting.htm.
- Meta-Llama. “Llama-Models/Models/Llama3_3/MODEL_CARD.md at Main · Meta-Llama/Llama-Models.” 2025. GitHub. 2025. https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md.
- Wu, Jet. 2025 “Jw-Source/LlamaSim: Simulate Human Behavior with Mass LLMs.” GitHub. 2025. https://github.com/jw-source/LlamaSim.
- Rozado, David. 2024. “The Political Preferences of LLMs.” PLoS ONE 19 (7): e0306621–21. https://doi.org/10.1371/journal.pone.0306621.
