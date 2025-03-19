from agent import Agent
from groq import Groq
import random
import time
from dotenv import load_dotenv
import os
from collections import Counter

load_dotenv()


class Network:
    """
    Initialize the Network with agents and necessary configuration.

    Args:
        num_agents (int): The total number of agents (senators).
        max_context_size (int): Maximum size of the conversation context.
        names (list): List of senator full names.
        identities (list): List of dictionaries containing senator identities.
    """
    def __init__(self, num_agents:int, max_context_size:int, names:list, identities:list):
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.names = names
        self.identities = identities
        self.num_agents = num_agents
        self.agents = self._init_agents()
        self.shared_context = []
        self.conversation_logs = []
        self.max_context_size = max_context_size

    def _init_agents(self):
        """
        Initialize the agents (senators) by creating Agent objects from names and identities.
        """
        start_time = time.time()
        agents = [Agent(name, identity, self.client) 
                  for name, identity in zip(self.names, self.identities)]
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"Generated {self.num_agents} agents in: {elapsed_time} seconds")
        return agents
    
    def _manage_context_size(self):
        """
        Manage the conversation context size by removing older messages if the total size exceeds max context size.
        """
        total_length = sum(len(msg) for msg in self.shared_context)
        while total_length > self.max_context_size:
            removed_msg = self.shared_context.pop(0)
            total_length -= len(removed_msg)

    def group_chat(self, prompt:str, chat_type:str, max_rounds:int):
        """
        Simulate a group chat (debate) where each agent participates based on the shared context.

        Args:
            prompt (str): The topic of the conversation.
            chat_type (str): The type of debate ('round_robin' or 'random').
            max_rounds (int): The number of rounds to simulate.

        Returns:
            conversation_logs (list): Logs of all agents' responses during the conversation.
        """
        round_count = 0
        while round_count < max_rounds: 
            if chat_type == "round_robin":
                for _, agent in enumerate(self.agents):
                    agent.prompt = prompt
                    agent_response = agent.chat(self.shared_context)
                    self.shared_context.append(agent.name + ": " + agent_response)
                    self.conversation_logs.append(agent.name + ": " + agent_response)
                    self._manage_context_size()
                    print(f"\n{agent.name}: {agent_response}")
            elif chat_type == "random":
                for _ in range(len(self.agents)):
                    agent = random.choice(self.agents)
                    agent.prompt = prompt
                    agent_response = agent.chat(self.shared_context)
                    self.shared_context.append(agent.name + ": " + agent_response)
                    self.conversation_logs.append(agent.name + ": " + agent_response)
                    self._manage_context_size()
                    print(f"\n{agent.name}: {agent_response}")
            round_count += 1
        return self.conversation_logs
    
    def predict(self, prompt:str, question:str):
        """
        Predict the voting behavior of the senators before and after the conversation.

        Args:
            prompt (str): The conversation prompt (bill).
            question (str): The specific voting question.

        Returns:
            one_output, zero_output, two_output (str): Voting results (percent change in Yea, Nay, and Maybe).
        """
        pre_choice = [None]*self.num_agents
        post_choice = [None]*self.num_agents

        print("\nInitial Votes (Before Debate):")
        for i, agent in enumerate(self.agents):
            pre_decision = int(agent.pre_predict(question))
            pre_choice[i] = pre_decision
            print(f"{agent.name}: {'Yea' if pre_decision == 1 else 'Nay' if pre_decision == 0 else 'Maybe'}")

        for i, agent in enumerate(self.agents):
            post_decision = int(agent.post_predict(prompt, question))
            post_choice[i] = post_decision

        print("\nFinal Votes (After Debate):")
        # Print final votes
        for i, agent in enumerate(self.agents):
            post_decision = post_choice[i]
            print(f"{agent.name}: {'Yea' if post_decision == 1 else 'Nay' if post_decision == 0 else 'Maybe'}")

        print(f"Pre-choice: {pre_choice}")
        print(f"Post-choice: {post_choice}")
        pre_choice_counts = Counter(pre_choice)
        post_choice_counts = Counter(post_choice)
        percent_increase_in_zeros = (post_choice_counts[0] - pre_choice_counts[0])/self.num_agents*100
        percent_increase_in_ones = (post_choice_counts[1] - pre_choice_counts[1])/self.num_agents*100
        percent_increase_in_twos = (post_choice_counts[2] - pre_choice_counts[2])/self.num_agents*100
        zero_output = ""
        one_output = ""
        two_output = ""
        if percent_increase_in_zeros>=0:
            zero_output = f"+{percent_increase_in_zeros}% No"
        else:
            zero_output = f"{percent_increase_in_zeros}% No"
        if percent_increase_in_ones>=0:
            one_output = f"+{percent_increase_in_ones}% Yes"
        else:
            one_output = f"{percent_increase_in_ones}% Yes"
        if percent_increase_in_twos>=0:   
            two_output = f"+{percent_increase_in_twos}% Maybe"
        else:
            two_output = f"{percent_increase_in_twos}% Maybe"
        print(one_output, zero_output, two_output)
        return one_output, zero_output, two_output
