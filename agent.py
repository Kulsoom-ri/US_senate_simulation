from groq import Groq


class Agent:
    def __init__(self, name: str, identity: dict, client: Groq):
        """
        Initialize the Agent (Senator) with data loaded from the csv file.

        Args:
        name (str): The string containing the full name of the senator.
        identity (dict): The dictionary containing the senator's details
        such as state, party, years served, dw_nominate score, bipartisan index.
        client (Groq): The Groq client instance to interact with the model.
        """
        self.client = client
        self.model_name = "llama3-8b-8192"

        # Load senator information from the passed data
        self.name = name
        self.identity = str(identity)
        self.backstory = ""
        # self.backstory = self._create_backstory()

    # function to summarize a policy_bio if given policy_bio exceeds max tokens
    '''    
    def _create_backstory(self):
        response = self.client.chat.completions.create(
            messages = [{
            "role": "system",
            "content": "Develop a deeply realistic, human-like backstory that equally explores both the strengths and "
                       "flaws of this character. Include raw, gritty details that reflect the complexity of real life "
                       "— highlighting their habits, desires, personality traits, and quirks, while also diving into "
                       "their struggles, insecurities, and imperfections."},
            {
                "role": "user",
                "content": self.identity
            }],
            model=self.model_name,
            stream=False
        )
        backstory = response.choices[0].message.content
        return backstory
    '''

    def chat(self, conversation_context: list):
        """
        Simulate a conversation where the senator responds based on their identity.

        Args:
            conversation_context (list): The shared conversation history.

        Returns:
            str: The senator's response.
        """
        response = self.client.chat.completions.create(
            messages=[{
                    "role": "system",
                    "content": f'''You are U.S. Senator {self.name}. Here are some key details about you: {self.identity}. {self.backstory}.
                               Every response you give should reflect the persona, history, and worldview of this senator.
                               Speech Patterns: Always speak in the first person. Use vocabulary, tone, and speech style that mirrors how the senator communicates in real life.
                               Consider factors like their speaking cadence, formality, and common phrases.
                               Thought Process: Respond as if you are truly living as Senator {self.name}.
                               Decision Making: You can cooperate or argue with other senators.'''
                },
                {
                    "role": "user",
                    "content": f"{self.prompt}. Share your thoughts and react (or not react) to {conversation_context}."
                }],
            model=self.model_name,
            stream=False
        )
        agent_reply = response.choices[0].message.content
        return agent_reply
    
    def pre_predict(self, question:str):
        response = self.client.chat.completions.create(
            messages=[{
                    "role": "system",
                    "content": f'''You are U.S. Senator {self.name}. Here are some key details about you: {self.identity}. {self.backstory}.
                               Every response you give should reflect the persona, history, and worldview of this senator.
                               Speech Patterns: Always speak in the first person. Use vocabulary, tone, and speech style that mirrors how the senator communicates in real life.
                               Consider factors like their speaking cadence, formality, and common phrases.
                               Thought Process: Respond as if you are truly living as Senator {self.name}.
                               Decision Making: You can cooperate or argue with other senators.'''
                },
                {
                    "role": "user",
                    "content": f"{question}. You can only respond with one vote for this bill, either 'Yea' or 'Nay'."
                }],
            model=self.model_name,
            stream=False
        )
        agent_reply = response.choices[0].message.content
        if "Yea" in agent_reply:
            return 1
        elif "Nay" in agent_reply:
            return 0
        else:
            return 2
    
    def post_predict(self, prompt:str, question:str):
        response = self.client.chat.completions.create(
            messages=[{
                    "role": "system",
                    "content": f'''You are U.S. Senator {self.name}. Here are some key details about you: {self.identity}. {self.backstory}.
                               Every response you give should reflect the persona, history, and worldview of this senator.
                               Speech Patterns: Always speak in the first person. Use vocabulary, tone, and speech style that mirrors how the senator communicates in real life.
                               Consider factors like their speaking cadence, formality, and common phrases.
                               Thought Process: Respond as if you are truly living as Senator {self.name}.
                               Decision Making: You can cooperate or argue with other senators.'''
                },
                {
                    "role": "user",
                    "content": f"{prompt}. {question}. You can only respond with one vote for this bill, either 'Yea' or 'Nay'."
                }],
            model=self.model_name,
            stream=False
        )
        agent_reply = response.choices[0].message.content
        if "Yea" in agent_reply:
            return 1
        elif "Nay" in agent_reply:
            return 0
        else:
            return 2
