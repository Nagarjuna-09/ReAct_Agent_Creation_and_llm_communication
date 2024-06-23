from langchain_core.callbacks import BaseCallbackHandler
from typing import Dict, Any, List

from langchain_core.outputs import LLMResult


# This new class is inherited from BaseCallbacHandler. This BaseCallbackHandler contains functions that can frecord all the
# interesting functions happening in langchain like calls to llm, responses from llm etc.,
# You can find it here https://python.langchain.com/v0.1/docs/modules/callbacks/
class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print(f"Prompt to that LLM was: \n {prompts[0]}")
        print("**********************")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print(f"LLM response: \n {response.generations[0][0].text}")
        print("**********************")
