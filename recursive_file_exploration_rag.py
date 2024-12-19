import os
from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal
from utils import State, display_app_graph, format_current_context, render_prompt, list_avaliable_relative_files, get_exploration_queue
from langgraph.graph import StateGraph, MessagesState, START, END
from file_interaction import FileInteractionAgent, DataReaderAgent, TextReaderAgent, ImageReaderAgent, NotebookReaderAgent
import json


class PydanticExploration(BaseModel):
    """
    ResponseFormatter is a model that structures the response for an exploration prompt.
    
    Example structure:

            {
                "explore": {
                    "datasource_path_1": {
                        "path/to/file_1.txt": "specific prompt for file_1.txt",
                        ...
                    },
                    ...
                },
                "give_final_answer": False
            }

    If exploration is not needed, the response should be:
    
            {
                "explore": {},
                "give_final_answer": True
            }

    Attributes:
        explore (Dict[str, Dict[str, str]]): A dictionary where the keys are datasource paths and the values are dictionaries.
            The inner dictionaries map file paths to their respective exploration prompts.
            
        give_final_answer (bool): A flag indicating whether the final answer should be given without further exploration.
    """
    explore: Optional[Dict[str, Dict[str, str]]] = Field(
        default_factory=dict,
        description="A dictionary where the keys are datasource paths and the values are dictionaries. The inner dictionaries map file paths to their respective exploration prompts."
    )
    give_final_answer: bool = Field(
        default=False,
        description="A flag indicating whether the final answer should be given without further exploration."
    )

class RFERag:
    """
    RFERag is a class designed for recursive file exploration using language models. It initializes with a language model, a structured language model, and a folder containing prompt templates. The class sets up a state graph workflow for file exploration and defines nodes and edges for the workflow. It also maps file types to their respective extensions and context reader agents.
        llm (BaseModel): The language model instance for text processing.
        structured_llm (BaseModel): The structured language model instance for data processing.
        prompts_folder (str): The absolute path to the folder containing prompt templates.
        workflow (StateGraph): The state graph workflow for file exploration.
        app (CompiledStateGraph): The compiled workflow application.
        file_extension_map (dict): A dictionary mapping file types to their respective extensions.
        context_agents_map (dict): A dictionary mapping file types to their respective context reader agents.
    Methods:
        __init__(llm: BaseModel, structured_llm: BaseModel, prompts_folder: str) -> None:
        get_agent(file_path: str) -> FileInteractionAgent:
            Returns the appropriate context reader agent based on the file extension.
        context_evaluation_node(state: State) -> State:
            Evaluates the context to determine if it is sufficient to answer the user prompt or if further exploration is needed.
        exploration_node(state: State) -> State:
            Explores the files in the exploration queue and returns the new context.
        update_context_node(state: State) -> State:
            Updates the context with the new information acquired from the exploration.
        give_answer_node(state: State) -> State:
            Provides the final answer to the user based on the accumulated context.
    """
    
    def __init__(   
            self,
            llm: BaseModel,
            structured_llm: BaseModel,
            prompts_folder: str,
            datasources_paths: list = [],
            max_exploration_counter: int = 3,
            max_explorations: int = 15
            ) -> None:
        """
        Initializes the recursive file exploration class.
        Args:
            llm: The language model to be used for text processing.
            structured_llm: The structured language model to be used for data processing.
            prompts_folder: The folder containing prompt templates.
        Attributes:
            llm: The language model instance.
            structured_llm: The structured language model instance.
            prompts_folder: The absolute path to the folder containing prompt templates.
            workflow: The state graph workflow for file exploration.
            app: The compiled workflow application.
            file_extension_map: A dictionary mapping file types to their respective extensions.
            context_agents_map: A dictionary mapping file types to their respective context reader agents.
        """
        self.llm = llm
        self.structured_llm = structured_llm

        self.prompts_folder = os.path.abspath(prompts_folder).replace("\\", "/") + "/"
        self.datasources = {}
        for path in datasources_paths:
            self.add_datasource(path)

        self.max_exploration_counter = max_exploration_counter
        self.max_explorations = max_explorations

        self.workflow = StateGraph(State)

        # Define the nodes
        self.workflow.add_node("context_evaluation", self.context_evaluation_node)
        self.workflow.add_node("exploration", self.exploration_node)
        self.workflow.add_node("update_context", self.update_context_node)
        self.workflow.add_node("give_final_answer", self.give_answer_node)

        # Define the edges
        self.workflow.add_edge(START, "context_evaluation")

        def decide_next_node(state: State) -> Literal["Files to explore", "Sufficient context", "Max explorations"]:
            if state["exploration_queue"] == []:
                return "Sufficient context"
            if state["exploration_counter"] >= self.max_exploration_counter:
                return "Max explorations"
            if state["num_explorations"] >= self.max_explorations:
                return "Max explorations"
            return "Files to explore"
        self.workflow.add_conditional_edges("context_evaluation", decide_next_node, path_map={
            "Files to explore": "exploration",
            "Sufficient context": "give_final_answer",
            "Max explorations": "give_final_answer"
        })

        self.workflow.add_edge("exploration", "update_context")
        self.workflow.add_edge("update_context", "context_evaluation")

        self.workflow.add_edge("give_final_answer", END)

        self.app = self.workflow.compile()

        self.file_extension_map = {
            "text": ["txt", "md", "py", "json", "html", "css", "js", "ts", "sql", "xml", "yml", "yaml", "log"],
            "data": ["csv", "parquet", "xlsx"],
            "image": ["png", "jpg", "jpeg"],
            "notebook": ["ipynb"]
        }

        self.context_agents_map = {
            "text": TextReaderAgent(self.llm, self.prompts_folder + "context_from_text_file.jinja2"),
            "data": DataReaderAgent(self.structured_llm, self.prompts_folder + "context_from_dataframe.jinja2"),
            "image": ImageReaderAgent(self.llm),
            "notebook": NotebookReaderAgent(self.llm, self.prompts_folder + "context_from_notebook_file.jinja2")
        }

        self.last_state = None

    def get_agent(self, file_path: str) -> FileInteractionAgent:
        file_extension = file_path.split(".")[-1]
        for context_type, extensions in self.file_extension_map.items():
            if file_extension in extensions:
                return self.context_agents_map[context_type]
        return None


    # class RecursiveFileExplorationRAG:
    # Main node of the loop, the file exploration node
    def context_evaluation_node(self, state: State) -> State:
        """
        This is the node that is responsible for the file exploration and also deciding if the context is enough to answer.
        This agent will receive the user prompt and the current context and will decide if the context is enough to answer the user prompt.
        If the context is not enough, then it will decide what files to explore to get the necessary information.
        """
        self.last_state = state

        if state.get("datasources", {}) == {}:
            # no datasources available
            return {}
        
        prompt_file = self.prompts_folder + "exploration_prompt.jinja2"
        current_context = format_current_context(state)
        main_prompt = state["main_prompt"]
        example_datasource_path = list(state["datasources"].keys())[0]
        
        prompt = render_prompt(prompt_file, {
            "current_context": current_context,
            "main_prompt": main_prompt,
            "example_datasource_path": example_datasource_path
        })

        try:
            answer = self.structured_llm.with_structured_output(PydanticExploration).invoke(prompt)
            if answer is None:
                # Some models have a problem with the structured output, I'm not sure why
                answer = self.structured_llm.invoke(prompt)
                answer = json.loads(answer.content)
                explore = answer.get("explore", {})
                give_final_answer = answer.get("give_final_answer", False)
            else:
                explore = answer.explore
                give_final_answer = answer.give_final_answer
                
            if give_final_answer:
                state["exploration_queue"] = []
                return state
            else:
                state["exploration_queue"] = get_exploration_queue(explore)
                return state
                
        except Exception as e:
            # case of error with the model response
            print(f"Error with the model response: {e}")
            state["exploration_queue"] = []
            return state

    def exploration_node(self, state: State) -> State:
        """
        This node is responsible for exploring the files and returning the new context.
        """
        self.last_state = state
        
        state["exploration_counter"] = state["exploration_counter"] + 1
        exploration_queue = state["exploration_queue"]
        if exploration_queue == []:
            return state
        
        aquired_context = []
        for exploration in exploration_queue:
            file_path, specific_prompt = exploration
            agent = self.get_agent(file_path)
            if agent is None:
                aquired_context.append((file_path, "Cannot explore this file, the file extension is not supported"))
                continue
            explored_files = state.get("explored_files", [])

            generated_context = agent.get_context_from_file(specific_prompt=specific_prompt, file_path=file_path, state=state)
            generated_context = "Prompt: " + specific_prompt + "\n" + generated_context
            # caso o arquivo já tivesse sido explorado, adicionar o aviso
            # de que o arquivo já foi explorado
            if file_path in explored_files:
                generated_context += "\nThis file has already been explored - Important: You should stop exploring the same files again!"
            else:
                explored_files.append(file_path)

            aquired_context.append((file_path, generated_context))

            state["num_explorations"] = state["num_explorations"] + 1
            state["explored_files"] = explored_files

        state["exploration_queue"] = aquired_context

        return state

    def update_context_node(self, state: State) -> State:
        """
        This node is responsible for updating the context with the new information aquired from the exploration.
        """
        self.last_state = state
        
        aquired_context = state["exploration_queue"]
        if aquired_context == []:
            return state

        incoming_context = "This is the context aquired from the exploration:\n\n"
        for relevant_content in aquired_context:
            file_path, generated_context = relevant_content
            incoming_context += f"""
            [[Start of notes on file: {file_path}]]
            {generated_context}
            [[End of notes on file: {file_path}]]
            """

        main_prompt = state["main_prompt"]
        project_structure = state["datasources"]
        current_context = state["dynamic_context"]

        prompt = render_prompt(self.prompts_folder + "update_internal_context.jinja2", {
            "main_prompt": main_prompt,
            "project_structure": project_structure,
            "current_context": current_context,
            "incoming_context": incoming_context
        })
        new_context = self.llm.invoke(prompt).content.strip()
        state["dynamic_context"] = new_context
        state["exploration_queue"] = []
        return state

    def give_answer_node(self, state: State) -> State:
        """
        This node is responsible for giving the final answer to the user.
        """
        self.last_state = state
        
        prompt_with_context = f"""
        {state['main_prompt']}

        [[Start of what you know about the project]]
        {format_current_context(state)}
        [[End of what you know about the project]]
        """

        final_answer = self.llm.invoke(prompt_with_context).content
        state["final_answer"] = final_answer
        self.last_state = state
        return state

    def display_app_graph(self) -> None:
        """
        Displays the state graph of the application.
        """
        display_app_graph(self.app)

    def add_datasource(self, source_path: str) -> None:
        """
        Adds a new data source to the datasources dictionary.

        This method takes a source path, lists all available relative files
        in that path, and stores them in the datasources dictionary with the
        source path as the key.

        Args:
            source_path (str): The path to the data source directory.

        Returns:
            None
        """
        source_path = os.path.abspath(source_path).replace("\\", "/") + "/"
        self.datasources[source_path] = list_avaliable_relative_files(source_path)


    def answer(self, prompt: str) -> Dict[str, str]:
        """
        Answers a given prompt by following the state graph workflow.
        """
        # initialize the state
        state = {
            "main_prompt": prompt,
            "dynamic_context": "No information about the project yet",
            "datasources": self.datasources,
            "exploration_queue": [],
            "final_answer": "",
            "exploration_counter": 0,
            "num_explorations": 0
        }
        state = State(**state)
        # run the application
        final_state = self.app.invoke(state)

        answer = final_state["final_answer"]
        context = format_current_context(final_state)

        return {
            "answer": answer,
            "context": context,
            "exploration_counter": final_state["exploration_counter"],
            "num_explorations": final_state["num_explorations"]
        }
