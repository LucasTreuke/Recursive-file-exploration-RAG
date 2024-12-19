from IPython.display import Image, display
from typing import TypedDict, Dict, List, Tuple
import jinja2
import os
from pydantic import BaseModel, Field
from typing import Literal
import base64

class State(TypedDict):
    """
    Attributes:
        main_prompt (str): The main prompt string used in the application.
        dynamic_context (str): The dynamic context string that can change.
        datasources (Dict[str, List[str]]): A dictionary containing various data sources and the files list for each.
        exploration_queue (List[Tuple[str, str]]): A list of tuples containing the path to the file and the prompt to be answered.
        final_answer (str): The final answer string to be displayed.
        exploration_counter (int): The current exploration counter.
        num_explorations (int): The total number of explorations.
        explored_files (List[str]): A list of explored files
    """
    main_prompt: str
    dynamic_context: str
    datasources: Dict[str, List[str]]
    exploration_queue: List[Tuple[str, str]]
    final_answer: str
    exploration_counter: int
    num_explorations: int
    explored_files: List[str]

def list_avaliable_relative_files(base_path: str) -> List[str]:
    """
    Creates a list with all relative paths to the base_path, e.g.:
    ["file_1.txt", "file_2.txt", "dir_1/file_3.txt", "dir_1/file_4.txt"]

    Args:
        base_path (str): The base directory path to start the file exploration.

    Returns:
        List[str]: A list of relative file paths.
    """
    base_path = os.path.abspath(base_path).replace("\\", "/")
    avaliable_files = []
    for root, dirs, files in os.walk(base_path):
        # Ignore directories that start with . or __
        dirs[:] = [d for d in dirs if not (d.startswith('.') or d.startswith('__'))]
        for file in files:
            if not (file.startswith('.') or file.startswith('__')):
                avaliable_files.append((root + "/" + file).replace(base_path, "").replace("\\", "/")[1::])
    return avaliable_files

def get_exploration_queue(exploration: Dict[str, Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    Creates a list of tuples containing each path to the file and the prompt to be answered.

    Args:
        datasources (Dict[str, Dict[str, str]]): A dictionary containing various data sources and the files + exploration prompts for each.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the path to the file and the prompt to be answered.
    """
    exploration_queue = []
    for datasource, files in exploration.items():
        for file, prompt in files.items():
            exploration_queue.append(((datasource + "/" + file).replace("//","/"), prompt))
    return exploration_queue

def render_prompt(prompt_path, prompt_variables):
    with open(prompt_path, encoding="utf-8") as f:
        template = jinja2.Template(f.read())
    return template.render(prompt_variables)


def format_current_context(state: State) -> str:
    """
    Formats the current context string.
    
    Args:
        state (State): The current state of the application.
        
    Returns:
        str: The formatted current context string.
    """
    datasources = str(state.get("datasources", "no datasources"))
    return f"""
    Avaliable datasources:
    '{datasources}'

    [[Start of Project notes]]
    {state.get('dynamic_context', '')}
    [[End of Project notes]]
    """

def display_app_graph(app):
    try:
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass    


# Formatação de resposta do modelo da avaliação
class EvaluationResponse(BaseModel):
    """
    EvaluationResponse is a model that structures the response for an evaluation prompt.
    """
    integralidade: Literal["Responde a todas as partes", "Responde parcialmente", "Não responde"] = Field(
        ...,
        description="Indicates whether the response addresses all parts of the question."
    )
    aderencia_a_chave_de_resposta: Literal["Sim", "Parcialmente", "Não"] = Field(
        ...,
        description="Indicates whether the response adheres to the provided answer key."
    )
    precisao_e_conformidade: Literal["Precisa", "Parcialmente precisa", "Imprecisa"] = Field(
        ...,
        description="Indicates the accuracy and correctness of the response."
    )
    informacoes_extras: Literal["Sem informações extras", "Corretas e relevantes", "Corretas, mas irrelevantes", "Incorretas"] = Field(
        ...,
        description="Indicates whether the response includes extra information and evaluates its correctness and relevance."
    )
    nota_final: Literal[0, 1, 2] = Field(
        ...,
        description="The final score attributed to the response based on the evaluation criteria."
    )
    justificativa_da_nota: str = Field(
        ...,
        description="A brief explanation justifying the assigned score, referencing the evaluation criteria."
    )


class Avaliador:
    def __init__(self, llm, base_conhecimento, prompt_path):
        self.llm = llm

        base_conhecimento = os.path.abspath(base_conhecimento).replace("\\", "/")
        with open(base_conhecimento, "r", encoding="utf-8") as file:
            self.base_conhecimento = file.read()
    
        self.prompt_path = os.path.abspath(prompt_path).replace("\\", "/")

    def avaliar(self, pergunta, resposta, chave_resposta):
        prompt = render_prompt(self.prompt_path, {
            "pergunta": pergunta,
            "resposta": resposta,
            "chave_de_resposta": chave_resposta,
            "base_de_conhecimento": self.base_conhecimento
        })
        answer = self.llm.with_structured_output(EvaluationResponse).invoke(prompt)
        return {
            "integralidade": answer.integralidade,
            "aderencia a chave de resposta": answer.aderencia_a_chave_de_resposta,
            "precisao e conformidade": answer.precisao_e_conformidade,
            "informacoes extras": answer.informacoes_extras,
            "nota final": answer.nota_final,
            "justificativa da nota": answer.justificativa_da_nota
        }
    
    def avaliar_linha(self, row):
        return self.avaliar(row["pergunta"], row["resposta"], row["chave de resposta"])
    
def image_to_base64(image_path: str) -> str:
    """
    Opens an image file and returns its Base64-encoded string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            # Read the binary data and encode it to Base64
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image
    except Exception as e:
        raise ValueError(f"Error while converting image to Base64: {str(e)}")