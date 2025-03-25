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

However, judgmental forecasting is time- and labor-intensive, prompting interest in automating the process using large language models (LLMs) since they are already trained on vast amounts of cross-domain data. LLMs forecast by integrating diverse textual data, reasoning through context, and generating probabilistic predictions, in comparison to traditional ML classifiers that rely on structured historical data. Halawi et al. (2024) demonstrate that a retrieval-augmented LLM-based forecasting system achieves a Brier score of 0.179 and an accuracy of 71.5%, coming close to the human crowd’s 0.149 Brier score and 77.0% accuracy (Brier scores are a measure of the accuracy of probabilistic predictions, calculated as the mean squared difference between a predicted probability and the actual outcome). The system excels when the crowd is uncertain and when more relevant articles are retrieved but struggles with over-hedging in high-certainty cases.

Similarly, across various disciplines studies have tested the predictive power of LLMs on diverse datasets, revealing both their strengths and limitations. In neuroscience, a study evaluted general-purpose LLMs against expert neuroscientists on BenchBrain, a benchmark designed for predicting neuroscience research outcomes. The general-purpose LLMs achieved 81% accuracy in predicting experimental outcomes and the fine-tuned model (BrainGPT) performed even better (Luo et al. 2024). Another study explored whether LLMs trained for language prediction could effectively forecast multivariate time series data, showing preliminary but definitive success in demand forecasting tasks, albeit with concerns about overfitting (Wolff et al. 2025).

Another study assessed LLMs on theory of mind (ToM) tasks i.e. evaluations designed to assess an individual's ability to understand and attribute mental states (like beliefs, desires, and intentions) to others. Eleven LLMs were tested on 40 false-belief tasks, a gold-standard benchmark for evaluating ToM capabilities. While older models failed entirely, GPT-3.5-turbo achieved 20% accuracy, and GPT-4 reached 75%, performing at the level of a six-year-old child. These results suggest that ToM-like reasoning may have emerged as an unintended byproduct of language model training (Kosinski 2024). This has significant implications for forecasting and decision-making, as improved reasoning capabilities could enhance LLMs’ ability to predict human behavior, market trends, and social dynamics. However, the study also underscores limitations- LLMs may still struggle with nuanced, open-ended reasoning, meaning their forecasts could lack depth or misinterpret complex scenarios. Moreover, if these capabilities emerge unintentionally, it raises concerns about unpredictability in decision-making applications. While LLMs show promise in assisting with forecasting, their results must be interpreted with caution.
  
### Can LLMs provide explainable predictions, and are they appropriate for causal reasoning?
Political-LLM notes that explanability of LLM outputs is essential as it ensures that results are interpretable and transparent, "fostering trust in politically sensitive applications". LLMs potentially offer novel tools for causal analysis by identifying patterns, modeling causal relationships, and generating counterfactual scenarios to explore "what-if" conditions. Explainability tools, including attention mechanisms, hypertuning parameters and prompt engineering, could enhance the transparency of LLM-driven causal analysis.

Despite these strengths, LLMs face limitations that challenge their reliability for causal reasoning. A study from 2023 presents the 'Generative AI Paradox' which states that "generative models seem to acquire generation abilities more effectively than understanding, in contrast to human intelligence where generation is usually harder." (West et al. 2023) The study tests whether language models (such as GPT4) truly understand their own generated content by asking them multiple-choice questions about it. The LLMs perform worse than humans in discerning their own generations as well as in answering questions about them (although only slightly worse on the latter with an average accuracy of almost 87%).

Model collapse, a degenerative process where models gradually lose touch with the true underlying data distribution when trained on AI-generated content, poses another challenge for predictive multi-agent frameworks (Shumailov et al. 2024). In such systems, where one model’s output serves as another’s input, errors and biases can compound over time, leading to degraded performance. This issue arises because generative models, without regular exposure to diverse, high-quality human-generated data, increasingly reinforce their own distortions rather than accurately reflecting real-world distributions. Addressing model collapse requires a combination of fine-tuning, data filtering, and reinforcement learning techniques to maintain model integrity and prevent systemic degradation.

Finally, the issue of hallucinations in LLMs poses a significant limitation for any predictive task. One study positions hallucinations as a structural issue by proving rigorously that: (1) no training dataset can achieve 100% completeness, thus guaranteeing the model will encounter unknown or contradictory information; (2) the "needle in a haystack" problem, or accurate information retrieval, is mathematically undecidable, meaning LLMs cannot deterministically locate correct facts within their data; (3) intent classification is also undecidable, preventing accurate interpretation of user prompts; (4) the LLM halting problem is undecidable, rendering the length and content of generated outputs unpredictable; and (5) even with fact-checking, the inherent limitations of LLMs prevent complete elimination of hallucinations. (Banerjee, Agarwal, and Singla 2024)

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
- *The Cook Political Report (“The 2023 Cook Partisan Voting Index (Cook PVISM)” 2023):* Introduced in 1997, the Cook Partisan Voting Index (Cook PVI) measures how each state and district performs at the presidential level compared to the nation as a whole. The dataset was utilized to get the 2023 Cook Partisan voting score for the state of each senator. The 2023 PVI scores were claculated using 2016 and 2020 presidential election results. The Cook PVI scores indicate a state's partisan lean by comparing its average Democratic or Republican performance to the national average, expressed in terms such as D+5 (Democratic-leaning by 5 points) or R+3 (Republican-leaning by 3 points). A score of EVEN signifies a state that votes in line with the national average.

**For floor votes:**

A total of 45 floor votes were selected for simulation, including votes on motions such as "On Motion to Discharge Committee," "On Overriding the Veto," "On Passage of the Bill," and "On Cloture on the Motion to Proceed." These votes were randomly chosen from a curated list comprising CQ Key Votes, which are identified by CQ's editorial staff as the most significant floor votes in determining legislative outcomes, as well as key pieces of legislation highlighted on Wikipedia and the most-viewed bills on Congress.gov. The selection included bills originating in both the House and Senate, as well as joint resolutions, but was limited to votes that involved at least one roll call vote in the Senate.
- *Congress.gov (“Congress.gov | Library of Congress” 2025):* This webstie was utilized for obtaining contextual information about each bill, including its title, summary, number of co-sponsors, the party and name of the introducing senator, and its policy area. Bill summaries are authored by the Congressional Research Service ("CRS provides Congress with analysis that is authoritative, confidential, objective, and non-partisan.")
- *Senate.gov (“U.S. Senate: Roll Call Votes 118th Congress” 2024):* This website was utilized to obtain detailed records of senators' voting behavior during floor votes, including whether they voted "yea," "nay," were present, or did not vote.

### Target
The target variable being predicted is a senator's vote during a specific floor vote. This can take 4 values: Yea, Nay, Present or Not Voting. 

### Features
The features fed into the LLM simulation were:
1. #### For Senators:
   - name – Senator's name
   - age – Senator's age
   - religion – Senator's religious affiliation (or Unaffliated/Not Mentioned)
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
   - dw_nominate – Ideological score (-1 to +1, liberal to conservative) based on roll call votes
   - bipartisan_index – Measure of how frequently the senator works with the opposing party
   - backstory – Senator's biographical background

2. #### For Bills:
   - vote_date – Date of the floor vote
   - type_vote – Type of vote (e.g., Passage, Cloture)
   - measure_summary – Summary of the bill under consideration
   - sponsor – Name of the senator who introduced the bill
   - introduced_party – Political party affiliation of the sponsor
   - num_cosponsors – Number of senators who cosponsored the bill
   - previous_action – Summary of prior legislative actions related to the bil

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
- The target variable (bill outcome) was encoded as 1 for passed and 0 for rejected. All other vote types were excluded.
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

Temperature settings were also adjusted to optimize model performance. For debate simulation, a temperature of 0.4 was used to allow for a moderate degree of randomness, fostering a diversity of arguments and reasoning styles. For debate summary generation, a lower temperature of 0.2 was set to make the output more deterministic and fact-based. Additionally, a fixed random seed was used for the reproducibility of results across multiple simulation runs (although due to the inherent output varaibility, setting a random seed in LLMs does not always guarantee deterministic outputs).

## Methodology
### Correlations and Classifications
To explore the predictive capability of various features on voting behavior, correlations between different variables and the target outcome (vote) were analyzed. The goal was to identify the strongest predictors of legislative decision-making. A correlation matrix was constructed to examine the relationships between key numerical variables, and pairwise scatterplots were used to visualize patterns in the data.

To assess classification accuracy, 12 different statistical models on the dataset were tested. Each model was trained and evaluated using cross-validation as well as a train-test split to determine its predictive performance in classifying voting outcomes. The models included Naïve Bayes, k-Nearest Neighbors, Logistic Regression, Linear Support Vector Classification, Support Vector Classification, Decision Tree Classifier, Extra Tree Classifier, Extra Trees Classifier, AdaBoost Classifier, Random Forest Classifier, Perceptron and Multi-layer Perceptron (MLP) Classifier. These models allowed us to compare how forecasting through LLMs compares to predictions made through traditional classification methods.

### Basic Simulation (Without Prompt Engineering or Parameter Controls)
A baseline simulation was conducted in which the system prompt was kept minimal: "You are US senator {full_name}." Senators were only provided with a bill summary and asked to vote. No additional contextual information, deliberation, or parameter tuning was included in this stage. A total of 30 floor votes were simulated in this manner, focusing solely on passage votes.

### Simulation with Prompt Engineering and Parameter Controls
To improve the accuracy and realism of the simulation, additional prompt engineering was introduced in the next phase. This involved modifying the system prompt to incorporate factors such as political ideology and party alignment in the development of each agent. More context about the bill such as the date of the vote and previous action was also given. Specific parameters, such as temperature were controlled to have moderate variability in output.

### Simulation with prompt engineering, with controlling parameters and after 2 rounds of debate
The final simulation incorporated both prompt engineering and structured deliberation. Each senator engaged in two rounds of debate before casting their final vote. The purpose of this was to observe how deliberation influenced decision-making and whether the debate led to shifts in voting behavior. Additionally, the impact of debate on model accuracy was assessed by comparing predicted votes before and after discussion.

### Determining Variable Weights via Regression Analysis
To gain insight into the factors influencing prediction accuracy across various simulations, multiple regression models were conducted. The dependent variable in these models was the overall prediction accuracy for each bill. Independent variables included various bill characteristics, such as the number of cosponsors, the required majority threshold, and the status of the election season. This analysis facilitated an understanding of the relative importance of each feature in shaping the outputs generated by the large language model (LLM). However, to comprehensively understand the reasoning behind the LLM's outputs, it is essential to fine-tune the model or request explanations for its predictions.

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

### Basic Simulation

### Advanced Simulation

## Discussion
### Significance of findings for Political Science
### Significance of findings for Computer Science
### Limitations
### Future Work

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
