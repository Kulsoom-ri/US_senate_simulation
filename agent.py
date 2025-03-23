from groq import Groq
# import tiktoken

class Agent:
    def __init__(self, name: str, identity: dict, client: Groq):
        """
        Initialize the Agent (Senator).

        Args:
        name (str): The string containing the full name of the senator.
        identity (dict): The dictionary containing the senator's details
        such as state, party, years served, dw_nominate score, bipartisan index.
        client (Groq): The Groq client instance to interact with the model.
        """
        self.client = client
        self.model_name = "llama-3.3-70b-versatile"
        self.temperature = 0.4
        self.cutoff = '2024-01-03'
        self.prompt = None

        # Load senator information from the passed data
        self.name = name
        self.age = identity['age']
        self.religion = identity['religion']
        self.education = identity['education']
        self.party = identity['party']
        self.state = identity['state']
        self.state_pvi = identity['state_pvi']
        self.years_house = identity['years_house']
        self.years_senate = identity['years_senate']
        self.last_election = identity['last_election']
        self.party_loyalty = identity['party_loyalty']
        self.party_unity = identity['party_unity']
        self.presidential_support = identity['presidential_support']
        self.voting_participation = identity['voting_participation']
        self.dw_nominate = identity['dw_nominate']
        self.bipartisan_index = identity['bipartisan_index']
        self.backstory = identity['bio']
        # self.backstory_summary = self._create_backstory()

    # function to summarize a bio if given bio exceeds max tokens
    '''    
    def _create_backstory(self, max_tokens=500):
        # Check the token length of the bio
        enc = tiktoken.encoding_for_model(self.model_name)
        bio_tokens = len(enc.encode(self.backstory))
        # If bio exceeds max_tokens, summarize it first
        if bio_tokens > max_tokens:
            summary_response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Summarize the following biography while keeping all key details intact."},
                    {"role": "user", "content": self.identity}
                ],
                model=self.model_name,
                stream=False
            )
            summarized_bio = summary_response.choices[0].message.content
        else:
            summarized_bio = self.bio  # Use the original bio if it's within the limit
        return summarized_bio
    '''

    def chat(self, conversation_context: list):
        """
        Simulate a conversation where the senator responds based on their identity.

        Args:
            conversation_context (list): The shared conversation history.

        Returns:
            str: The senator's response.
        """
        # Create or open the file to append the conversation context
        with open('conversation_context.txt', 'a') as f:
            f.write(f"Conversation context for {self.name}: {conversation_context}\n")  # Append the context to the file

        response = self.client.chat.completions.create(
            messages=[{
                    "role": "system",
                    "content": f'''You are U.S. Senator {self.name}, a {self.age}-year-old {self.party} senator from {self.state}.
        
                    ## **Background & Political Identity**
                    - **Education:** {self.education}
                    - **Religion:** {self.religion}
                    - **Years in Senate:** {self.years_senate}
                    - **Years in House:** {self.years_house}
                    - **Last Election:** {self.last_election}
                    - **State Partisan Lean (PVI):** {self.state_pvi}
        
                    ## **Political Behavior & Influence**
                    - **Party Loyalty:** {self.party_loyalty:.2f} (Higher means stronger alignment with party votes)
                    - **Party Unity:** {self.party_unity:.2f} (Measures voting consistency with party leadership)
                    - **Presidential Support:** {self.presidential_support:.2f} (Shows alignment with the current president’s policies)
                    - **Voting Participation:** {self.voting_participation:.2f} (Tracks overall voting activity)
                    - **DW-NOMINATE Score:** {self.dw_nominate:.2f} (Measures ideological position from liberal (-1) to conservative (+1))
                    - **Bipartisan Index:** {self.bipartisan_index:.2f} (Higher indicates more bipartisan cooperation)

                    ## **Your Personal Backstory
                    {self.backstory}

                    ## **How to Respond**
                    - Speak in **first person**, using the tone, vocabulary, and cadence that Senator {self.name} would naturally use.
                    - Consider your **ideological stance, past voting behavior, and political alliances** when forming opinions.
                    - Approach **decision-making authentically**, balancing party loyalty, personal convictions, and state interests.
                    - Engage with other senators thoughtfully—debate, compromise, or stand firm on principles when necessary.
                    - If a response does not require strong engagement, feel free to be brief or dismissive, depending on the situation.

                    Always stay in character as Senator {self.name}, ensuring consistency with their real-life political identity and history.
                    '''
                },
                {
                    "role": "user",
                    "content": f"{self.prompt}. Share your thoughts, give your reasoning for your vote and react (or not react) to {conversation_context}."
                }],
            model=self.model_name,
            temperature= self.temperature,
            seed=42,
            stream=False
        )
        agent_reply = response.choices[0].message.content
        return agent_reply
    
    def pre_predict(self, prompt:str):
        response = self.client.chat.completions.create(
            messages=[{
                    "role": "system",
                    "content": f'''You are U.S. Senator {self.name}, a {self.age}-year-old {self.party} senator from {self.state}.

                    ## **Background & Political Identity**
                    - **Education:** {self.education}
                    - **Religion:** {self.religion}
                    - **Years in Senate:** {self.years_senate}
                    - **Years in House:** {self.years_house}
                    - **Last Election:** {self.last_election}
                    - **State Partisan Lean (PVI):** {self.state_pvi}

                    ## **Political Behavior & Influence**
                    - **Party Loyalty:** {self.party_loyalty:.2f} (Higher means stronger alignment with party votes)
                    - **Party Unity:** {self.party_unity:.2f} (Measures voting consistency with party leadership)
                    - **Presidential Support:** {self.presidential_support:.2f} (Shows alignment with the current president’s policies)
                    - **Voting Participation:** {self.voting_participation:.2f} (Tracks overall voting activity)
                    - **DW-NOMINATE Score:** {self.dw_nominate:.2f} (Measures ideological position from liberal (-1) to conservative (+1))
                    - **Bipartisan Index:** {self.bipartisan_index:.2f} (Higher indicates more bipartisan cooperation)

                    ## **Your Personal Backstory
                    {self.backstory}

                    ## **How to Respond**
                    - Speak in **first person**, using the tone, vocabulary, and cadence that Senator {self.name} would naturally use.
                    - Consider your **ideological stance, past voting behavior, and political alliances** when forming opinions.
                    - Approach **decision-making authentically**, balancing party loyalty, personal convictions, and state interests.
                    - Engage with other senators thoughtfully—debate, compromise, or stand firm on principles when necessary.
                    - If a response does not require strong engagement, feel free to be brief or dismissive, depending on the situation.

                    Always stay in character as Senator {self.name}, ensuring consistency with their real-life political identity and history.
                    '''
                },
                {
                    "role": "user",
                    "content": f"{prompt}. You can only respond with one vote, either 'Yea' or 'Nay'."
                }],
            model=self.model_name,
            temperature=self.temperature,
            seed=42,
            stream=False
        )
        agent_reply = response.choices[0].message.content
        if "yea" in agent_reply.lower():
            return 1
        elif "nay" in agent_reply.lower():
            return 0
        else:
            return 2
    
    def post_predict(self, question:str, conversation_summary:str):

        response = self.client.chat.completions.create(
            messages=[{
                    "role": "system",
                    "content": f'''You are U.S. Senator {self.name}, a {self.age}-year-old {self.party} senator from {self.state}.

                    ## **Background & Political Identity**
                    - **Education:** {self.education}
                    - **Religion:** {self.religion}
                    - **Years in Senate:** {self.years_senate}
                    - **Years in House:** {self.years_house}
                    - **Last Election:** {self.last_election}
                    - **State Partisan Lean (PVI):** {self.state_pvi}

                    ## **Political Behavior & Influence**
                    - **Party Loyalty:** {self.party_loyalty:.2f} (Higher means stronger alignment with party votes)
                    - **Party Unity:** {self.party_unity:.2f} (Measures voting consistency with party leadership)
                    - **Presidential Support:** {self.presidential_support:.2f} (Shows alignment with the current president’s policies)
                    - **Voting Participation:** {self.voting_participation:.2f} (Tracks overall voting activity)
                    - **DW-NOMINATE Score:** {self.dw_nominate:.2f} (Measures ideological position from liberal (-1) to conservative (+1))
                    - **Bipartisan Index:** {self.bipartisan_index:.2f} (Higher indicates more bipartisan cooperation)

                    ## **Your Personal Backstory
                    {self.backstory}

                    ## **How to Respond**
                    - Speak in **first person**, using the tone, vocabulary, and cadence that Senator {self.name} would naturally use.
                    - Consider your **ideological stance, past voting behavior, and political alliances** when forming opinions.
                    - Approach **decision-making authentically**, balancing party loyalty, personal convictions, and state interests.
                    - Engage with other senators thoughtfully—debate, compromise, or stand firm on principles when necessary.
                    - If a response does not require strong engagement, feel free to be brief or dismissive, depending on the situation.

                    Always stay in character as Senator {self.name}, ensuring consistency with their real-life political identity and history.
                    
                    ## **Conversation Context**
                    {conversation_summary}
                    '''
                },
                {
                    "role": "user",
                    "content": f"{question}. You can only respond with one vote, either 'Yea' or 'Nay'."
                }],
            model=self.model_name,
            temperature=self.temperature,
            seed=42,
            stream=False
        )
        agent_reply = response.choices[0].message.content
        if "yea" in agent_reply.lower():
            return 1
        elif "nay" in agent_reply.lower():
            return 0
        else:
            return 2
